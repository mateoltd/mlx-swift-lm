import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

final class Qwen35PipelineTests: XCTestCase {
    func testShardedLayerTypesCreateMatchingCaches() throws {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(
                """
                {
                  "model_type": "qwen3_5_text",
                  "num_hidden_layers": 8,
                  "full_attention_interval": 4,
                  "hidden_size": 128,
                  "intermediate_size": 256,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 2,
                  "head_dim": 32,
                  "linear_num_value_heads": 4,
                  "linear_num_key_heads": 2,
                  "linear_key_head_dim": 32,
                  "linear_value_head_dim": 32,
                  "vocab_size": 256
                }
                """.utf8
            )
        )
        let model = Qwen35TextModel(configuration)
        let shard = Array(model.model.layers[2 ..< 6])

        model.model.replaceLayers(shard, shardOffset: 2)

        XCTAssertEqual(model.model.layerIsLinear, [true, false, true, true])

        let caches = model.newCache(parameters: GenerateParameters(kvBits: 8))
        XCTAssertEqual(caches.count, 4)
        XCTAssertTrue(caches[0] is MambaCache)
        XCTAssertTrue(caches[1] is QuantizedKVCache)
        XCTAssertTrue(caches[2] is MambaCache)
        XCTAssertTrue(caches[3] is MambaCache)
    }
}
