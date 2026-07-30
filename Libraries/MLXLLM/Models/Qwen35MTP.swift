// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Qwen35MTPConfiguration: Codable, Sendable {
    public let modelType: String
    public let textConfiguration: Qwen35TextConfiguration
    public let blockSize: Int
    public let mtpHiddenLayers: Int

    private struct MTPTextConfiguration: Decodable {
        let base: Qwen35TextConfiguration
        let mtpHiddenLayers: Int

        private enum CodingKeys: String, CodingKey {
            case mtpHiddenLayers = "mtp_num_hidden_layers"
        }

        init(from decoder: Decoder) throws {
            base = try Qwen35TextConfiguration(from: decoder)
            let values = try decoder.container(keyedBy: CodingKeys.self)
            mtpHiddenLayers =
                try values.decodeIfPresent(Int.self, forKey: .mtpHiddenLayers) ?? 1
        }
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case blockSize = "block_size"
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try values.decodeIfPresent(String.self, forKey: .modelType)
            ?? "qwen3_5_mtp"
        let text = try values.decode(
            MTPTextConfiguration.self, forKey: .textConfiguration)
        textConfiguration = text.base
        mtpHiddenLayers = max(1, text.mtpHiddenLayers)
        blockSize =
            try values.decodeIfPresent(Int.self, forKey: .blockSize)
            ?? (mtpHiddenLayers + 2)
    }
}

/// Native Qwen3.5/3.6 MTP sidecar.
///
/// The sidecar owns one full-attention decoder layer and a compact private KV
/// history. It borrows the verified target's embeddings and LM head, so its
/// proposals never alter the target distribution.
public final class Qwen35MTPDraftModel: Module, MTPDrafterModel {
    @ModuleInfo(key: "fc") private var fc: Linear
    @ModuleInfo(key: "pre_fc_norm_embedding") private var preFCNormEmbedding: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") private var preFCNormHidden: RMSNorm

    private var layers: [Qwen35DecoderLayer]
    private let norm: RMSNorm

    public let configuration: Qwen35MTPConfiguration

    private var cache: [KVCache] = []
    private var seedToken: MLXArray?
    private var seedHidden: MLXArray?
    private var prefillCarryHidden: MLXArray?
    private var roundAppended = 0

    public var requiresTargetHistoryPrefill: Bool { true }

    public init(_ configuration: Qwen35MTPConfiguration) {
        self.configuration = configuration
        let text = configuration.textConfiguration

        _fc.wrappedValue = Linear(text.hiddenSize * 2, text.hiddenSize, bias: false)
        _preFCNormEmbedding.wrappedValue = RMSNorm(
            dimensions: text.hiddenSize, eps: text.rmsNormEps)
        _preFCNormHidden.wrappedValue = RMSNorm(
            dimensions: text.hiddenSize, eps: text.rmsNormEps)

        var layerConfiguration = text
        layerConfiguration.hiddenLayers = configuration.mtpHiddenLayers
        layerConfiguration.fullAttentionInterval = 1
        self.layers = (0 ..< configuration.mtpHiddenLayers).map {
            Qwen35DecoderLayer(layerConfiguration, layerIdx: $0)
        }
        self.norm = RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        super.init()
    }

    public func reset(target _: any LanguageModel) {
        cache.removeAll(keepingCapacity: false)
        seedToken = nil
        seedHidden = nil
        prefillCarryHidden = nil
        roundAppended = 0
    }

    private func targetTextModel(_ target: any LanguageModel) -> Qwen35TextModel? {
        if let text = target as? Qwen35TextModel {
            return text
        }
        if let wrapped = target as? Qwen35Model {
            return wrapped.languageModel
        }
        return nil
    }

    private func distributedTransport(
        _ target: any LanguageModel
    ) -> (rank: Int, worldSize: Int, group: DistributedGroup)? {
        targetTextModel(target)?.model.pipelineTransport
    }

    private func targetLogits(
        _ hidden: MLXArray, target: any LanguageModel
    ) -> MLXArray {
        guard let target = targetTextModel(target) else {
            preconditionFailure(
                "Qwen35MTPDraftModel requires a Qwen35TextModel or Qwen35Model target")
        }
        if let lmHead = target.lmHead {
            return lmHead(hidden)
        }
        return target.model.embedTokens.asLinear(hidden)
    }

    private func forwardTokens(
        _ token: MLXArray,
        hidden: MLXArray,
        target: any LanguageModel
    ) -> MLXArray {
        guard let target = targetTextModel(target) else {
            preconditionFailure(
                "Qwen35MTPDraftModel requires a Qwen35TextModel or Qwen35Model target")
        }

        let tokenEmbedding = target.model.embedTokens(token)
        var h = concatenated(
            [preFCNormEmbedding(tokenEmbedding), preFCNormHidden(hidden)],
            axis: -1
        )
        h = fc(h)

        for (index, layer) in layers.enumerated() {
            let layerCache = cache[index]
            h = layer(
                h,
                attentionMask: createAttentionMask(h: h, cache: layerCache),
                ssmMask: nil,
                cache: layerCache
            )
        }
        return norm(h)
    }

    public func prefillTargetHistory(
        target: any LanguageModel,
        tokens: MLXArray,
        targetHidden: MLXArray,
        startPosition: Int
    ) {
        if cache.isEmpty {
            cache = (0 ..< layers.count).map { _ in
                PositionedKVCache(basePosition: 0)
            }
        }

        let count = tokens.size
        guard count > 0 else { return }

        let pairedTokens: MLXArray
        let pairedHidden: MLXArray
        if startPosition == 0 {
            guard count > 1 else {
                prefillCarryHidden = targetHidden[0..., (-1)..., 0...]
                eval(prefillCarryHidden!)
                return
            }
            pairedTokens = tokens[1...][.newAxis]
            pairedHidden = targetHidden[0..., ..<(count - 1), 0...]
        } else {
            guard let carry = prefillCarryHidden else {
                preconditionFailure("Qwen MTP prompt chunks must be contiguous")
            }
            pairedTokens = tokens[.newAxis]
            let within =
                count > 1
                ? targetHidden[0..., ..<(count - 1), 0...]
                : targetHidden[0..., ..<0, 0...]
            pairedHidden = concatenated([carry, within], axis: 1)
        }

        _ = forwardTokens(pairedTokens, hidden: pairedHidden, target: target)
        eval(cache.flatMap(\.state))
        prefillCarryHidden = targetHidden[0..., (-1)..., 0...]
        eval(prefillCarryHidden!)
    }

    private func sampledToken(
        _ hidden: MLXArray,
        target: any LanguageModel,
        sampler: any LogitSampler
    ) -> MLXArray {
        let logits = targetLogits(hidden, target: target)[0..., -1, 0...]
        return sampler.sample(logits: logits)[0..., .newAxis]
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        if cache.isEmpty {
            cache = (0 ..< layers.count).map { _ in
                PositionedKVCache(basePosition: queryOffset)
            }
        }

        var token = lastToken[.newAxis]
        var hidden = lastHidden
        var proposed: [MLXArray] = []
        roundAppended = 0

        if let primedToken = seedToken, let primedHidden = seedHidden {
            token = primedToken
            hidden = primedHidden
            proposed.append(token)
            seedToken = nil
            seedHidden = nil
        }

        while proposed.count < blockSize - 1 {
            hidden = forwardTokens(token, hidden: hidden, target: target)
            roundAppended += 1
            token = sampledToken(hidden, target: target, sampler: sampler)
            proposed.append(token)
        }

        var result = concatenated(proposed, axis: 1)
        if let transport = distributedTransport(target), transport.worldSize == 2 {
            precondition(
                transport.rank == 0,
                "The weighted Qwen MTP drafter must only run on pipeline rank 0")
            result = transport.group.send(result, dest: 1, stream: .cpu)
            result.eval()
        }
        return result
    }

    public func acceptVerifiedTokens(
        target: any LanguageModel,
        verifyHidden: MLXArray,
        draftTokens _: MLXArray,
        accepted: Int,
        bonusToken: MLXArray,
        sampler: any LogitSampler
    ) {
        let kept = min(max(0, accepted), roundAppended)
        let rejectedAppends = roundAppended - kept
        if rejectedAppends > 0 {
            for layerCache in cache {
                _ = layerCache.trim(rejectedAppends)
            }
        }

        // Pair the verifier's correction/next bonus with the true target
        // hidden at that position. The resulting prediction seeds the first
        // draft of the next round without another sidecar forward.
        let slot = min(max(0, accepted), verifyHidden.dim(1) - 1)
        let committedHidden = verifyHidden[0..., slot ..< (slot + 1), 0...]
        let committedToken = bonusToken[.newAxis]
        let nextHidden = forwardTokens(
            committedToken, hidden: committedHidden, target: target)
        seedHidden = nextHidden
        seedToken = sampledToken(nextHidden, target: target, sampler: sampler)
        roundAppended = 0
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (originalKey, value) in weights {
            let key =
                originalKey.hasPrefix("mtp.")
                ? String(originalKey.dropFirst("mtp.".count))
                : originalKey
            sanitized[key] = value
        }
        return sanitized
    }
}

/// Weightless pipeline worker. Rank 0 owns the 475 MB sidecar and sends each
/// proposal block once; the iPhone receives exactly those tokens before
/// entering the same target-verification collective sequence.
public final class Qwen35MTPFollowerModel: Module, MTPDrafterModel {
    public var requiresTargetHistoryPrefill: Bool { true }

    public override init() {
        super.init()
    }

    private func targetTextModel(_ target: any LanguageModel) -> Qwen35TextModel? {
        if let text = target as? Qwen35TextModel {
            return text
        }
        if let wrapped = target as? Qwen35Model {
            return wrapped.languageModel
        }
        return nil
    }

    public func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        queryOffset _: Int,
        blockSize: Int,
        sampler _: any LogitSampler
    ) -> MLXArray {
        guard let transport = targetTextModel(target)?.model.pipelineTransport,
            transport.worldSize == 2,
            transport.rank == 1
        else {
            preconditionFailure(
                "Qwen35MTPFollowerModel requires pipeline rank 1 of a two-rank target")
        }
        let received = transport.group.recvLike(
            MLXArray.zeros(
                [1, blockSize - 1],
                dtype: lastToken.dtype
            ),
            source: 0,
            stream: .cpu
        )
        received.eval()
        return received
    }
}

public enum Qwen35MTPRegistration {
    public static func register() async {
        await MTPDrafterTypeRegistry.shared.registerModelType(
            "qwen3_5_mtp",
            creator: { data in
                let configuration = try JSONDecoder().decode(
                    Qwen35MTPConfiguration.self, from: data)
                return Qwen35MTPDraftModel(configuration)
            }
        )
    }
}
