package com.jaredgary.hn.monitoring;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cache.CacheManager;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.stats.CacheStats;

/**
 * Performance Monitor for tracking application metrics.
 * 
 * @author Gary Gonzalez Zepeda
 * @version 2.0.0
 */
@Component
public class PerformanceMonitor {

    private static final Logger LOGGER = LogManager.getLogger(PerformanceMonitor.class);

    @Autowired
    private CacheManager cacheManager;

    /**
     * Log cache statistics every 5 minutes.
     */
    @Scheduled(fixedDelay = 300000) // 5 minutes
    public void logCacheStatistics() {
        LOGGER.info("=== Cache Performance Statistics ===");
        
        cacheManager.getCacheNames().forEach(cacheName -> {
            org.springframework.cache.Cache cache = cacheManager.getCache(cacheName);
            if (cache != null) {
                // Try to get native cache for statistics
                Object nativeCache = cache.getNativeCache();
                if (nativeCache instanceof Cache) {
                    Cache<?, ?> caffeineCache = (Cache<?, ?>) nativeCache;
                    CacheStats stats = caffeineCache.stats();
                    
                    LOGGER.info("Cache '{}' - Hit Rate: {:.2f}%, Hits: {}, Misses: {}, Evictions: {}", 
                               cacheName, 
                               stats.hitRate() * 100,
                               stats.hitCount(),
                               stats.missCount(),
                               stats.evictionCount());
                } else {
                    LOGGER.info("Cache '{}' - Statistics not available", cacheName);
                }
            }
        });
    }

    /**
     * Log application health every 10 minutes.
     */
    @Scheduled(fixedDelay = 600000) // 10 minutes
    public void logApplicationHealth() {
        Runtime runtime = Runtime.getRuntime();
        long maxMemory = runtime.maxMemory();
        long totalMemory = runtime.totalMemory();
        long freeMemory = runtime.freeMemory();
        long usedMemory = totalMemory - freeMemory;
        
        double memoryUsagePercent = (double) usedMemory / maxMemory * 100;
        
        LOGGER.info("=== Application Health ===");
        LOGGER.info("Memory Usage: {:.2f}% ({} MB / {} MB)", 
                   memoryUsagePercent,
                   usedMemory / 1024 / 1024,
                   maxMemory / 1024 / 1024);
        LOGGER.info("Available Processors: {}", runtime.availableProcessors());
        
        if (memoryUsagePercent > 80) {
            LOGGER.warn("High memory usage detected: {:.2f}%", memoryUsagePercent);
        }
    }
}