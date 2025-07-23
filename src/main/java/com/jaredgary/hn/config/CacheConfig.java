package com.jaredgary.hn.config;

import java.util.concurrent.TimeUnit;

import org.springframework.cache.CacheManager;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.cache.caffeine.CaffeineCacheManager;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.github.benmanes.caffeine.cache.Caffeine;

/**
 * Cache Configuration for optimized performance.
 * 
 * @author Gary Gonzalez Zepeda
 * @version 2.0.0
 */
@Configuration
@EnableCaching
public class CacheConfig {

    /**
     * Configure Caffeine Cache Manager for high performance caching.
     *
     * @return CacheManager instance
     */
    @Bean
    public CacheManager cacheManager() {
        CaffeineCacheManager cacheManager = new CaffeineCacheManager();
        cacheManager.setCaffeine(Caffeine.newBuilder()
                .maximumSize(1000)
                .expireAfterAccess(5, TimeUnit.MINUTES)
                .expireAfterWrite(10, TimeUnit.MINUTES)
                .recordStats());
        return cacheManager;
    }

    /**
     * Custom Caffeine configuration for products cache.
     *
     * @return Caffeine instance
     */
    @Bean
    public Caffeine<Object, Object> caffeineConfig() {
        return Caffeine.newBuilder()
                .maximumSize(500)
                .expireAfterAccess(5, TimeUnit.MINUTES)
                .recordStats();
    }
}