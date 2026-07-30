// Copyright © 2026 Apple Inc.

import MLX

/// A regular attention cache whose rotary position can begin after an
/// uncached prefix.
///
/// Qwen's external MTP head may start with an empty private attention cache
/// while the verified target is already thousands of tokens into a request.
/// Storage and masks must still describe only the drafter's private history,
/// but RoPE must use the target's absolute position.
public final class PositionedKVCache: BaseKVCache {
    public let basePosition: Int
    private let storage: KVCacheSimple

    public init(basePosition: Int) {
        self.basePosition = max(0, basePosition)
        self.storage = KVCacheSimple()
        super.init()
    }

    private init(basePosition: Int, storage: KVCacheSimple) {
        self.basePosition = basePosition
        self.storage = storage
        super.init()
        self.offset = storage.offset
    }

    public override var ropeOffset: RoPEOffset {
        .scalar(basePosition + offset)
    }

    public override func innerState() -> [MLXArray] {
        storage.state
    }

    public override func update(
        keys: MLXArray, values: MLXArray
    ) -> (MLXArray, MLXArray) {
        let result = storage.update(keys: keys, values: values)
        offset = storage.offset
        return result
    }

    public override var state: [MLXArray] {
        get { storage.state }
        set {
            storage.state = newValue
            offset = storage.offset
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = storage.trim(n)
        offset = storage.offset
        return trimmed
    }

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        storage.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    public override func copy() -> any KVCache {
        let copied = KVCacheSimple()
        if !storage.state.isEmpty {
            copied.state = storage.state.map { $0[.ellipsis] }
        }
        return PositionedKVCache(basePosition: basePosition, storage: copied)
    }
}
