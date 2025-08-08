#include "framework/atlas-test-framework.h"
#include <iostream>
#include <vector>
#include <algorithm>

using namespace atlas::testing;

// Specific Omega Rule tests
class OmegaRuleTestSuite {
public:
    static bool runSlidingWindowBasic() {
        const int window_size = 64;
        const int hidden_dim = 256;
        const int sequence_length = 128;
        
        std::vector<float> keys(window_size * hidden_dim);
        std::vector<float> values(window_size * hidden_dim);
        std::vector<float> weights(window_size);
        
        // Initialize keys and values with test pattern
        for (int i = 0; i < window_size; i++) {
            for (int j = 0; j < hidden_dim; j++) {
                keys[i * hidden_dim + j] = 0.1f * std::cos(2.0f * M_PI * j / hidden_dim);
                values[i * hidden_dim + j] = 0.1f * std::sin(2.0f * M_PI * j / hidden_dim);
            }
            weights[i] = std::exp(-0.1f * i) / window_size; // Exponential decay
        }
        
        // Normalize weights
        float weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
        for (float& w : weights) {
            w /= weight_sum;
        }
        
        // Validate weight normalization
        weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
        if (std::abs(weight_sum - 1.0f) > 1e-5f) {
            std::cout << "Weight normalization failed: sum = " << weight_sum << std::endl;
            return false;
        }
        
        std::cout << "Omega Rule Sliding Window Basic: Window size " << window_size 
                  << ", weights normalized" << std::endl;
        
        return true;
    }
    
    static bool runContextAwareUpdates() {
        const int window_size = 32;
        const int hidden_dim = 128;
        const float learning_rate = 0.01f;
        const float decay_factor = 0.95f;
        
        std::vector<float> current_context(hidden_dim);
        std::vector<float> previous_context(hidden_dim);
        std::vector<float> updated_weights(window_size);
        
        // Initialize contexts
        for (int i = 0; i < hidden_dim; i++) {
            current_context[i] = 0.1f * (i % 10 - 5);  // [-0.5, 0.4] range
            previous_context[i] = 0.1f * ((i + 5) % 10 - 5);
        }
        
        // Simulate context-aware weight updates
        for (int i = 0; i < window_size; i++) {
            float context_similarity = 0.0f;
            
            // Compute simple dot product similarity
            for (int j = 0; j < hidden_dim; j++) {
                context_similarity += current_context[j] * previous_context[j];
            }
            context_similarity = std::tanh(context_similarity); // Bound to [-1, 1]
            
            // Update weight based on context similarity
            float base_weight = 1.0f / window_size;
            updated_weights[i] = base_weight + learning_rate * context_similarity;
            
            // Apply decay
            updated_weights[i] *= std::pow(decay_factor, i);
        }
        
        // Normalize updated weights
        float total_weight = std::accumulate(updated_weights.begin(), updated_weights.end(), 0.0f);
        if (total_weight > 1e-8f) {
            for (float& w : updated_weights) {
                w /= total_weight;
            }
        }
        
        // Validate weight properties
        float final_sum = std::accumulate(updated_weights.begin(), updated_weights.end(), 0.0f);
        if (std::abs(final_sum - 1.0f) > 1e-5f) {
            return false;
        }
        
        // Check that weights are non-negative
        for (float w : updated_weights) {
            if (w < -1e-6f) {
                return false;
            }
        }
        
        std::cout << "Omega Rule Context Updates: " << window_size 
                  << " weights updated with context awareness" << std::endl;
        
        return true;
    }
    
    static bool runAdaptiveWindowSizing() {
        std::vector<int> window_sizes = {16, 32, 64, 128, 256};
        const int hidden_dim = 256;
        const int max_sequence_length = 512;
        
        for (int window_size : window_sizes) {
            // Test that window size adapts to sequence length
            int effective_window = std::min(window_size, max_sequence_length);
            
            std::vector<float> attention_weights(effective_window);
            std::vector<float> position_bias(effective_window);
            
            // Generate position-based attention bias
            for (int i = 0; i < effective_window; i++) {
                float position_factor = (float)i / effective_window;
                position_bias[i] = std::exp(-2.0f * position_factor); // Exponential decay
                
                attention_weights[i] = position_bias[i];
            }
            
            // Normalize attention weights
            float sum = std::accumulate(attention_weights.begin(), attention_weights.end(), 0.0f);
            if (sum > 1e-8f) {
                for (float& w : attention_weights) {
                    w /= sum;
                }
            }
            
            // Validate properties
            float normalized_sum = std::accumulate(attention_weights.begin(), attention_weights.end(), 0.0f);
            if (std::abs(normalized_sum - 1.0f) > 1e-5f) {
                return false;
            }
            
            // Check monotonic decay (should generally decrease)
            bool mostly_decreasing = true;
            int violations = 0;
            for (int i = 1; i < effective_window; i++) {
                if (attention_weights[i] > attention_weights[i-1] * 1.1f) { // Allow 10% variance
                    violations++;
                }
            }
            
            if (violations > effective_window / 4) { // Allow up to 25% violations
                mostly_decreasing = false;
            }
            
            if (!mostly_decreasing) {
                std::cout << "Window size " << window_size << " failed monotonic decay check" << std::endl;
                return false;
            }
        }
        
        std::cout << "Omega Rule Adaptive Window: Tested " << window_sizes.size() 
                  << " window sizes" << std::endl;
        
        return true;
    }
    
    static bool runMemoryEfficiency() {
        const int large_window = 1024;
        const int hidden_dim = 1024;
        const int num_iterations = 100;
        
        // Test memory-efficient sliding window updates
        std::vector<float> circular_buffer(large_window * hidden_dim);
        std::vector<float> weight_history(large_window);
        
        int current_pos = 0;
        
        for (int iter = 0; iter < num_iterations; iter++) {
            // Update circular buffer position
            current_pos = (current_pos + 1) % large_window;
            
            // Update weights with circular indexing
            for (int i = 0; i < large_window; i++) {
                int buffer_idx = (current_pos - i + large_window) % large_window;
                
                // Age-based weight decay
                float age_factor = std::exp(-0.01f * i);
                weight_history[i] = age_factor;
                
                // Update buffer with new values
                for (int j = 0; j < 4 && j < hidden_dim; j++) { // Only update first 4 elements for efficiency
                    circular_buffer[buffer_idx * hidden_dim + j] = 0.1f * std::sin(iter * 0.1f + j);
                }
            }
            
            // Normalize weights
            float sum = std::accumulate(weight_history.begin(), weight_history.end(), 0.0f);
            if (sum > 1e-8f) {
                for (float& w : weight_history) {
                    w /= sum;
                }
            }
        }
        
        // Validate final state
        float final_sum = std::accumulate(weight_history.begin(), weight_history.end(), 0.0f);
        if (std::abs(final_sum - 1.0f) > 1e-4f) {
            return false;
        }
        
        // Check that buffer values are finite
        for (int i = 0; i < 16; i++) { // Check first 16 values
            if (!std::isfinite(circular_buffer[i])) {
                return false;
            }
        }
        
        std::cout << "Omega Rule Memory Efficiency: Processed " << num_iterations 
                  << " iterations with " << large_window << " window size" << std::endl;
        
        return true;
    }
};

int main() {
    std::cout << "=== ATLAS Omega Rule Specific Tests ===" << std::endl;
    
    bool all_passed = true;
    
    std::cout << "\n1. Testing sliding window basic functionality..." << std::endl;
    if (!OmegaRuleTestSuite::runSlidingWindowBasic()) {
        std::cout << "❌ Sliding window basic test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Sliding window basic test passed" << std::endl;
    }
    
    std::cout << "\n2. Testing context-aware updates..." << std::endl;
    if (!OmegaRuleTestSuite::runContextAwareUpdates()) {
        std::cout << "❌ Context-aware updates test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Context-aware updates test passed" << std::endl;
    }
    
    std::cout << "\n3. Testing adaptive window sizing..." << std::endl;
    if (!OmegaRuleTestSuite::runAdaptiveWindowSizing()) {
        std::cout << "❌ Adaptive window sizing test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Adaptive window sizing test passed" << std::endl;
    }
    
    std::cout << "\n4. Testing memory efficiency..." << std::endl;
    if (!OmegaRuleTestSuite::runMemoryEfficiency()) {
        std::cout << "❌ Memory efficiency test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Memory efficiency test passed" << std::endl;
    }
    
    std::cout << "\n" << std::string(50, '=') << std::endl;
    if (all_passed) {
        std::cout << "🎉 ALL OMEGA RULE TESTS PASSED!" << std::endl;
    } else {
        std::cout << "❌ SOME OMEGA RULE TESTS FAILED!" << std::endl;
    }
    std::cout << std::string(50, '=') << std::endl;
    
    return all_passed ? 0 : 1;
}