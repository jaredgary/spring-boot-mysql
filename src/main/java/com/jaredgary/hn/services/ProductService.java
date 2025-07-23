package com.jaredgary.hn.services;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.CachePut;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.jaredgary.hn.model.Product;
import com.jaredgary.hn.repositories.ProductRepository;

@Service
@Transactional(readOnly = true)
public class ProductService {

    private static final Logger LOGGER = LogManager.getLogger(ProductService.class);

    @Autowired
    private ProductRepository repository;

    /**
     * Get all products with caching
     *
     * @return List of all products
     */
    @Cacheable(value = "products", key = "'all'")
    public List<Product> listAll() {
        LOGGER.info("Fetching all products from database");
        return repository.findAll();
    }

    /**
     * Get products with pagination
     *
     * @param pageable Pagination information
     * @return Page of products
     */
    @Cacheable(value = "products", key = "'page_' + #pageable.pageNumber + '_' + #pageable.pageSize")
    public Page<Product> listAll(Pageable pageable) {
        LOGGER.info("Fetching products with pagination: page {}, size {}", 
                   pageable.getPageNumber(), pageable.getPageSize());
        return repository.findAll(pageable);
    }

    /**
     * Get product by ID with caching
     *
     * @param id Product ID
     * @return Product or null if not found
     */
    @Cacheable(value = "products", key = "#id")
    public Product getById(Long id) {
        LOGGER.info("Fetching product with ID: {}", id);
        Optional<Product> product = repository.findById(id);
        return product.orElse(null);
    }

    /**
     * Search products by description
     *
     * @param description Description to search for
     * @return List of matching products
     */
    @Cacheable(value = "products", key = "'search_' + #description")
    public List<Product> searchByDescription(String description) {
        LOGGER.info("Searching products by description: {}", description);
        return repository.findByDescriptionContaining(description);
    }

    /**
     * Find products by price range
     *
     * @param minPrice Minimum price
     * @param maxPrice Maximum price
     * @return List of products in price range
     */
    @Cacheable(value = "products", key = "'price_range_' + #minPrice + '_' + #maxPrice")
    public List<Product> findByPriceRange(Double minPrice, Double maxPrice) {
        LOGGER.info("Finding products in price range: {} - {}", minPrice, maxPrice);
        return repository.findByPriceBetween(minPrice, maxPrice);
    }

    /**
     * Save or update product with cache management
     *
     * @param product Product to save
     * @return Saved product
     */
    @Transactional(readOnly = false)
    @CachePut(value = "products", key = "#result.id")
    @CacheEvict(value = "products", key = "'all'")
    public Product saveOrUpdate(Product product) {
        LOGGER.info("Saving/updating product: {}", product.getDescription());
        Product savedProduct = repository.save(product);
        LOGGER.info("Product saved with ID: {}", savedProduct.getId());
        return savedProduct;
    }

    /**
     * Save multiple products in batch
     *
     * @param products List of products to save
     * @return List of saved products
     */
    @Transactional(readOnly = false)
    @CacheEvict(value = "products", allEntries = true)
    public List<Product> saveAll(List<Product> products) {
        LOGGER.info("Batch saving {} products", products.size());
        List<Product> savedProducts = repository.saveAll(products);
        LOGGER.info("Batch save completed for {} products", savedProducts.size());
        return savedProducts;
    }

    /**
     * Delete product by ID with cache eviction
     *
     * @param id Product ID to delete
     */
    @Transactional(readOnly = false)
    @CacheEvict(value = "products", allEntries = true)
    public void delete(Long id) {
        LOGGER.info("Deleting product with ID: {}", id);
        repository.deleteById(id);
        LOGGER.info("Product deleted successfully");
    }

    /**
     * Check if product exists
     *
     * @param id Product ID
     * @return true if exists, false otherwise
     */
    public boolean existsById(Long id) {
        return repository.existsById(id);
    }

    /**
     * Get total count of products
     *
     * @return Total count
     */
    @Cacheable(value = "products", key = "'count'")
    public long getProductCount() {
        return repository.countProducts();
    }

    /**
     * Update prices asynchronously
     *
     * @param ids List of product IDs
     * @param multiplier Price multiplier
     * @return CompletableFuture for async processing
     */
    @Async
    @Transactional(readOnly = false)
    @CacheEvict(value = "products", allEntries = true)
    public CompletableFuture<Void> updatePricesAsync(List<Long> ids, Double multiplier) {
        LOGGER.info("Updating prices for {} products with multiplier {}", ids.size(), multiplier);
        repository.updatePricesByIds(ids, multiplier);
        LOGGER.info("Price update completed for {} products", ids.size());
        return CompletableFuture.completedFuture(null);
    }

    /**
     * Complex search with multiple criteria
     *
     * @param description Description to search
     * @param minPrice Minimum price
     * @param maxPrice Maximum price
     * @return List of matching products
     */
    public List<Product> findByMultipleCriteria(String description, Double minPrice, Double maxPrice) {
        LOGGER.info("Complex search: description='{}', price range: {}-{}", 
                   description, minPrice, maxPrice);
        return repository.findByDescriptionContainingAndPriceBetween(description, minPrice, maxPrice);
    }
}
