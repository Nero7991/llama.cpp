#include "atlas-server.hpp"
#include <algorithm>
#include <numeric>

namespace atlas {

// Extended metrics implementation for detailed monitoring
namespace metrics {

    // Performance counter for high-resolution timing
    class high_resolution_counter {
    private:
        std::chrono::high_resolution_clock::time_point start_;
        
    public:
        high_resolution_counter() : start_(std::chrono::high_resolution_clock::now()) {}
        
        double elapsed_microseconds() const {
            auto now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(now - start_).count();
        }
        
        double elapsed_milliseconds() const {
            return elapsed_microseconds() / 1000.0;
        }
        
        void reset() {
            start_ = std::chrono::high_resolution_clock::now();
        }
    };

    // Sliding window for calculating moving averages
    template<typename T>
    class sliding_window {
    private:
        std::deque<T> values_;
        size_t max_size_;
        T sum_;
        
    public:
        explicit sliding_window(size_t max_size) : max_size_(max_size), sum_{} {}
        
        void add_value(T value) {
            values_.push_back(value);
            sum_ += value;
            
            if (values_.size() > max_size_) {
                sum_ -= values_.front();
                values_.pop_front();
            }
        }
        
        T average() const {
            return values_.empty() ? T{} : sum_ / static_cast<T>(values_.size());
        }
        
        T median() const {
            if (values_.empty()) return T{};
            
            std::vector<T> sorted(values_.begin(), values_.end());
            std::sort(sorted.begin(), sorted.end());
            
            size_t mid = sorted.size() / 2;
            if (sorted.size() % 2 == 0) {
                return (sorted[mid - 1] + sorted[mid]) / 2;
            } else {
                return sorted[mid];
            }
        }
        
        T percentile(double p) const {
            if (values_.empty()) return T{};
            
            std::vector<T> sorted(values_.begin(), values_.end());
            std::sort(sorted.begin(), sorted.end());
            
            double index = p * (sorted.size() - 1);
            size_t lower = static_cast<size_t>(std::floor(index));
            size_t upper = static_cast<size_t>(std::ceil(index));
            
            if (lower == upper) {
                return sorted[lower];
            } else {
                double weight = index - lower;
                return sorted[lower] * (1 - weight) + sorted[upper] * weight;
            }
        }
        
        T min() const {
            return values_.empty() ? T{} : *std::min_element(values_.begin(), values_.end());
        }
        
        T max() const {
            return values_.empty() ? T{} : *std::max_element(values_.begin(), values_.end());
        }
        
        size_t size() const { return values_.size(); }
        bool empty() const { return values_.empty(); }
        void clear() { values_.clear(); sum_ = T{}; }
    };

    // Advanced metrics collector with detailed statistics
    class advanced_metrics_collector {
    private:
        mutable std::mutex metrics_mutex_;
        
        // Basic counters
        std::atomic<uint64_t> total_requests_{0};
        std::atomic<uint64_t> successful_requests_{0};
        std::atomic<uint64_t> failed_requests_{0};
        std::atomic<uint64_t> active_requests_{0};
        
        // Latency tracking
        sliding_window<double> latency_window_{1000};
        std::atomic<double> min_latency_{std::numeric_limits<double>::max()};
        std::atomic<double> max_latency_{0.0};
        
        // Throughput tracking
        sliding_window<double> throughput_window_{60}; // 60 second window
        std::chrono::steady_clock::time_point last_throughput_calc_;
        
        // Cache statistics
        std::atomic<uint64_t> cache_hits_{0};
        std::atomic<uint64_t> cache_misses_{0};
        sliding_window<double> cache_hit_ratio_window_{100};
        
        // Memory usage tracking
        std::atomic<uint64_t> current_memory_usage_{0};
        std::atomic<uint64_t> peak_memory_usage_{0};
        sliding_window<uint64_t> memory_usage_window_{60};
        
        // Error tracking
        std::unordered_map<std::string, uint64_t> error_counts_;
        std::deque<std::pair<std::chrono::steady_clock::time_point, std::string>> recent_errors_;
        
        // System resource tracking
        std::atomic<double> cpu_usage_{0.0};
        std::atomic<double> memory_usage_percent_{0.0};
        
        // Collection control
        std::atomic<bool> collecting_{false};
        std::thread collection_thread_;
        std::chrono::steady_clock::time_point start_time_;
        
    public:
        advanced_metrics_collector() : 
            last_throughput_calc_(std::chrono::steady_clock::now()),
            start_time_(std::chrono::steady_clock::now()) {}
        
        ~advanced_metrics_collector() {
            stop_collection();
        }
        
        void start_collection(int update_interval_ms = 1000) {
            if (collecting_.load()) return;
            
            collecting_ = true;
            collection_thread_ = std::thread([this, update_interval_ms]() {
                collect_system_metrics(update_interval_ms);
            });
        }
        
        void stop_collection() {
            collecting_ = false;
            if (collection_thread_.joinable()) {
                collection_thread_.join();
            }
        }
        
        // Request lifecycle tracking
        void record_request_start() {
            total_requests_++;
            active_requests_++;
        }
        
        void record_request_end(double latency_ms, bool success, const std::string& error_type = "") {
            active_requests_--;
            
            if (success) {
                successful_requests_++;
            } else {
                failed_requests_++;
                record_error(error_type.empty() ? "unknown_error" : error_type);
            }
            
            // Update latency statistics
            {
                std::lock_guard<std::mutex> lock(metrics_mutex_);
                latency_window_.add_value(latency_ms);
            }
            
            // Update min/max latency atomically
            double current_min = min_latency_.load();
            while (latency_ms < current_min && 
                   !min_latency_.compare_exchange_weak(current_min, latency_ms)) {
                // Retry until successful or value is no longer smaller
            }
            
            double current_max = max_latency_.load();
            while (latency_ms > current_max && 
                   !max_latency_.compare_exchange_weak(current_max, latency_ms)) {
                // Retry until successful or value is no longer larger
            }
        }
        
        // Cache tracking
        void record_cache_hit() {
            cache_hits_++;
            update_cache_hit_ratio();
        }
        
        void record_cache_miss() {
            cache_misses_++;
            update_cache_hit_ratio();
        }
        
        // Memory tracking
        void update_memory_usage(uint64_t bytes) {
            current_memory_usage_ = bytes;
            
            uint64_t current_peak = peak_memory_usage_.load();
            while (bytes > current_peak && 
                   !peak_memory_usage_.compare_exchange_weak(current_peak, bytes)) {
                // Retry until successful or value is no longer larger
            }
            
            {
                std::lock_guard<std::mutex> lock(metrics_mutex_);
                memory_usage_window_.add_value(bytes);
            }
        }
        
        // Error tracking
        void record_error(const std::string& error_type) {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            error_counts_[error_type]++;
            recent_errors_.emplace_back(std::chrono::steady_clock::now(), error_type);
            
            // Keep only recent errors (last 100)
            while (recent_errors_.size() > 100) {
                recent_errors_.pop_front();
            }
        }
        
        // Comprehensive metrics report
        nlohmann::json get_comprehensive_metrics() const {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            
            auto now = std::chrono::steady_clock::now();
            auto uptime_seconds = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
            
            // Calculate throughput
            double total_completed = successful_requests_.load() + failed_requests_.load();
            double throughput_rps = uptime_seconds > 0 ? total_completed / uptime_seconds : 0.0;
            
            // Calculate success rate
            double success_rate = total_requests_.load() > 0 ? 
                static_cast<double>(successful_requests_.load()) / total_requests_.load() : 0.0;
            
            // Cache hit ratio
            uint64_t total_cache_requests = cache_hits_.load() + cache_misses_.load();
            double cache_hit_ratio = total_cache_requests > 0 ? 
                static_cast<double>(cache_hits_.load()) / total_cache_requests : 0.0;
            
            return {
                {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count()},
                {"uptime_seconds", uptime_seconds},
                
                // Request statistics
                {"requests", {
                    {"total", total_requests_.load()},
                    {"active", active_requests_.load()},
                    {"successful", successful_requests_.load()},
                    {"failed", failed_requests_.load()},
                    {"success_rate", success_rate}
                }},
                
                // Latency statistics
                {"latency_ms", {
                    {"min", min_latency_.load()},
                    {"max", max_latency_.load()},
                    {"avg", latency_window_.empty() ? 0.0 : latency_window_.average()},
                    {"median", latency_window_.empty() ? 0.0 : latency_window_.median()},
                    {"p95", latency_window_.empty() ? 0.0 : latency_window_.percentile(0.95)},
                    {"p99", latency_window_.empty() ? 0.0 : latency_window_.percentile(0.99)}
                }},
                
                // Throughput statistics
                {"throughput", {
                    {"requests_per_second", throughput_rps},
                    {"avg_rps_1min", throughput_window_.empty() ? 0.0 : throughput_window_.average()}
                }},
                
                // Cache statistics
                {"cache", {
                    {"hits", cache_hits_.load()},
                    {"misses", cache_misses_.load()},
                    {"hit_ratio", cache_hit_ratio},
                    {"avg_hit_ratio", cache_hit_ratio_window_.empty() ? 0.0 : cache_hit_ratio_window_.average()}
                }},
                
                // Memory statistics
                {"memory", {
                    {"current_bytes", current_memory_usage_.load()},
                    {"peak_bytes", peak_memory_usage_.load()},
                    {"avg_bytes", memory_usage_window_.empty() ? 0 : memory_usage_window_.average()},
                    {"usage_percent", memory_usage_percent_.load()}
                }},
                
                // System statistics
                {"system", {
                    {"cpu_usage_percent", cpu_usage_.load()},
                    {"memory_usage_percent", memory_usage_percent_.load()}
                }},
                
                // Error statistics
                {"errors", {
                    {"total_types", error_counts_.size()},
                    {"recent_count", recent_errors_.size()},
                    {"by_type", error_counts_}
                }}
            };
        }
        
        // Performance summary for quick monitoring
        nlohmann::json get_performance_summary() const {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            
            double success_rate = total_requests_.load() > 0 ? 
                static_cast<double>(successful_requests_.load()) / total_requests_.load() : 0.0;
            
            uint64_t total_cache_requests = cache_hits_.load() + cache_misses_.load();
            double cache_hit_ratio = total_cache_requests > 0 ? 
                static_cast<double>(cache_hits_.load()) / total_cache_requests : 0.0;
            
            return {
                {"active_requests", active_requests_.load()},
                {"success_rate", success_rate},
                {"avg_latency_ms", latency_window_.empty() ? 0.0 : latency_window_.average()},
                {"cache_hit_ratio", cache_hit_ratio},
                {"memory_usage_mb", current_memory_usage_.load() / (1024 * 1024)},
                {"healthy", success_rate >= 0.95 && active_requests_.load() < 100}
            };
        }
        
        // Reset all metrics
        void reset_metrics() {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            
            total_requests_ = 0;
            successful_requests_ = 0;
            failed_requests_ = 0;
            active_requests_ = 0;
            
            latency_window_.clear();
            min_latency_ = std::numeric_limits<double>::max();
            max_latency_ = 0.0;
            
            throughput_window_.clear();
            cache_hit_ratio_window_.clear();
            memory_usage_window_.clear();
            
            cache_hits_ = 0;
            cache_misses_ = 0;
            
            current_memory_usage_ = 0;
            peak_memory_usage_ = 0;
            
            error_counts_.clear();
            recent_errors_.clear();
            
            start_time_ = std::chrono::steady_clock::now();
        }
        
    private:
        void update_cache_hit_ratio() {
            uint64_t hits = cache_hits_.load();
            uint64_t misses = cache_misses_.load();
            uint64_t total = hits + misses;
            
            if (total > 0) {
                double ratio = static_cast<double>(hits) / total;
                std::lock_guard<std::mutex> lock(metrics_mutex_);
                cache_hit_ratio_window_.add_value(ratio);
            }
        }
        
        void collect_system_metrics(int interval_ms) {
            while (collecting_.load()) {
                // Update throughput
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_throughput_calc_);
                
                if (elapsed.count() >= 1) {
                    double completed_in_period = successful_requests_.load() + failed_requests_.load();
                    double rps = completed_in_period / elapsed.count();
                    
                    {
                        std::lock_guard<std::mutex> lock(metrics_mutex_);
                        throughput_window_.add_value(rps);
                    }
                    
                    last_throughput_calc_ = now;
                }
                
                // Collect system resource usage (simplified)
                // In a real implementation, this would query actual system stats
                cpu_usage_ = 50.0; // Placeholder
                memory_usage_percent_ = 60.0; // Placeholder
                
                std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
            }
        }
    };

} // namespace metrics

} // namespace atlas