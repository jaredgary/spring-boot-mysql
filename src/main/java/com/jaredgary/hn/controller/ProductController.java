package com.jaredgary.hn.controller;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import javax.validation.Valid;
import javax.validation.constraints.Min;
import javax.validation.constraints.Positive;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import com.jaredgary.hn.constants.Constants;
import com.jaredgary.hn.model.Product;
import com.jaredgary.hn.services.ProductService;

import io.micrometer.core.annotation.Timed;

/**
 * ProductController - Optimized REST API for Product management.
 *
 * @author Gary Gonzalez Zepeda <mailto:gary.gonzalez@tigo.com.hn />
 * @version 2.0.0
 * @since 07-15-2019 01:56:52 PM 2019
 */
@RestController
@RequestMapping("/api/v1/products")
@Validated
@CrossOrigin(origins = "*", maxAge = 3600)
public class ProductController {

    /** Attribute that determine a Constant of LOGGER. */
    private static final Logger LOGGER = LogManager.getLogger(ProductController.class);

    /** Attribute that determine service. */
    @Autowired
    private ProductService service;

    /**
     * List all products with pagination support.
     *
     * @param page Page number (0-based)
     * @param size Page size
     * @param sortBy Sort field
     * @param sortDir Sort direction (asc/desc)
     * @return Paginated response entity with products
     */
    @GetMapping
    @Timed(value = "products.list.time", description = "Time taken to list products")
    public ResponseEntity<Page<Product>> listAll(
            @RequestParam(defaultValue = "0") @Min(0) int page,
            @RequestParam(defaultValue = "10") @Min(1) int size,
            @RequestParam(defaultValue = "id") String sortBy,
            @RequestParam(defaultValue = "asc") String sortDir) {
        
        LOGGER.info("Listing products - page: {}, size: {}, sortBy: {}, sortDir: {}", 
                   page, size, sortBy, sortDir);
        
        Sort.Direction direction = sortDir.equalsIgnoreCase("desc") ? 
            Sort.Direction.DESC : Sort.Direction.ASC;
        Pageable pageable = PageRequest.of(page, size, Sort.by(direction, sortBy));
        
        Page<Product> products = service.listAll(pageable);
        
        LOGGER.info("Retrieved {} products from database", products.getTotalElements());
        return ResponseEntity.ok(products);
    }

    /**
     * Get all products without pagination (for backward compatibility).
     *
     * @return List of all products
     */
    @GetMapping("/all")
    @Timed(value = "products.list.all.time", description = "Time taken to list all products")
    public ResponseEntity<List<Product>> listAllProducts() {
        LOGGER.info("Listing all products without pagination");
        List<Product> products = service.listAll();
        return ResponseEntity.ok(products);
    }

    /**
     * Gets product by ID.
     *
     * @param id the product ID
     * @return the product
     */
    @GetMapping("/{id}")
    @Timed(value = "products.get.time", description = "Time taken to get product by ID")
    public ResponseEntity<Product> getById(@PathVariable @Positive Long id) {
        LOGGER.info("Fetching product with ID: {}", id);
        
        Product product = service.getById(id);
        if (product == null) {
            LOGGER.warn("Product not found with ID: {}", id);
            return ResponseEntity.notFound().build();
        }
        
        LOGGER.info("Product found with ID: {}", id);
        return ResponseEntity.ok(product);
    }

    /**
     * Creates a new product.
     *
     * @param product the product to create
     * @return the created product
     */
    @PostMapping
    @Timed(value = "products.create.time", description = "Time taken to create product")
    public ResponseEntity<Product> createProduct(@Valid @RequestBody Product product) {
        LOGGER.info("Creating new product: {}", product.getDescription());
        
        // Ensure ID is null for new products
        product.setId(null);
        Product savedProduct = service.saveOrUpdate(product);
        
        LOGGER.info("Product created with ID: {}", savedProduct.getId());
        return ResponseEntity.status(HttpStatus.CREATED).body(savedProduct);
    }

    /**
     * Updates an existing product.
     *
     * @param id the product ID
     * @param product the product data
     * @return the updated product
     */
    @PutMapping("/{id}")
    @Timed(value = "products.update.time", description = "Time taken to update product")
    public ResponseEntity<Product> updateProduct(@PathVariable @Positive Long id, 
                                                @Valid @RequestBody Product product) {
        LOGGER.info("Updating product with ID: {}", id);
        
        if (!service.existsById(id)) {
            LOGGER.warn("Product not found for update with ID: {}", id);
            return ResponseEntity.notFound().build();
        }
        
        product.setId(id);
        Product updatedProduct = service.saveOrUpdate(product);
        
        LOGGER.info("Product updated with ID: {}", id);
        return ResponseEntity.ok(updatedProduct);
    }

    /**
     * Deletes a product.
     *
     * @param id the product ID
     * @return response entity with status
     */
    @DeleteMapping("/{id}")
    @Timed(value = "products.delete.time", description = "Time taken to delete product")
    public ResponseEntity<String> deleteProduct(@PathVariable @Positive Long id) {
        LOGGER.info("Deleting product with ID: {}", id);
        
        if (!service.existsById(id)) {
            LOGGER.warn("Product not found for deletion with ID: {}", id);
            return ResponseEntity.notFound().build();
        }
        
        service.delete(id);
        LOGGER.info("Product deleted with ID: {}", id);
        return ResponseEntity.ok(Constants.SUCCESS.getMessage());
    }

    /**
     * Search products by description.
     *
     * @param description the search term
     * @return list of matching products
     */
    @GetMapping("/search")
    @Timed(value = "products.search.time", description = "Time taken to search products")
    public ResponseEntity<List<Product>> searchProducts(@RequestParam String description) {
        LOGGER.info("Searching products by description: {}", description);
        
        List<Product> products = service.searchByDescription(description);
        
        LOGGER.info("Found {} products matching description: {}", products.size(), description);
        return ResponseEntity.ok(products);
    }

    /**
     * Find products by price range.
     *
     * @param minPrice minimum price
     * @param maxPrice maximum price
     * @return list of products in price range
     */
    @GetMapping("/price-range")
    @Timed(value = "products.price.range.time", description = "Time taken to find products by price range")
    public ResponseEntity<List<Product>> findByPriceRange(
            @RequestParam @Positive Double minPrice,
            @RequestParam @Positive Double maxPrice) {
        
        LOGGER.info("Finding products in price range: {} - {}", minPrice, maxPrice);
        
        if (minPrice > maxPrice) {
            LOGGER.warn("Invalid price range: min {} > max {}", minPrice, maxPrice);
            return ResponseEntity.badRequest().build();
        }
        
        List<Product> products = service.findByPriceRange(minPrice, maxPrice);
        
        LOGGER.info("Found {} products in price range: {} - {}", products.size(), minPrice, maxPrice);
        return ResponseEntity.ok(products);
    }

    /**
     * Complex search with multiple criteria.
     *
     * @param description description to search
     * @param minPrice minimum price
     * @param maxPrice maximum price
     * @return list of matching products
     */
    @GetMapping("/advanced-search")
    @Timed(value = "products.advanced.search.time", description = "Time taken for advanced search")
    public ResponseEntity<List<Product>> advancedSearch(
            @RequestParam(required = false, defaultValue = "") String description,
            @RequestParam(required = false, defaultValue = "0") @Min(0) Double minPrice,
            @RequestParam(required = false, defaultValue = "999999.99") @Positive Double maxPrice) {
        
        LOGGER.info("Advanced search: description='{}', price range: {}-{}", 
                   description, minPrice, maxPrice);
        
        List<Product> products = service.findByMultipleCriteria(description, minPrice, maxPrice);
        
        LOGGER.info("Advanced search returned {} products", products.size());
        return ResponseEntity.ok(products);
    }

    /**
     * Batch create products.
     *
     * @param products list of products to create
     * @return list of created products
     */
    @PostMapping("/batch")
    @Timed(value = "products.batch.create.time", description = "Time taken for batch create")
    public ResponseEntity<List<Product>> batchCreateProducts(@Valid @RequestBody List<Product> products) {
        LOGGER.info("Batch creating {} products", products.size());
        
        // Ensure all IDs are null for new products
        products.forEach(product -> product.setId(null));
        
        List<Product> savedProducts = service.saveAll(products);
        
        LOGGER.info("Batch created {} products", savedProducts.size());
        return ResponseEntity.status(HttpStatus.CREATED).body(savedProducts);
    }

    /**
     * Update prices asynchronously.
     *
     * @param ids list of product IDs
     * @param multiplier price multiplier
     * @return async response
     */
    @PatchMapping("/prices")
    @Timed(value = "products.update.prices.time", description = "Time taken to update prices")
    public ResponseEntity<String> updatePrices(@RequestParam List<Long> ids, 
                                              @RequestParam @Positive Double multiplier) {
        LOGGER.info("Updating prices for {} products with multiplier {}", ids.size(), multiplier);
        
        CompletableFuture<Void> future = service.updatePricesAsync(ids, multiplier);
        
        return ResponseEntity.accepted().body("Price update initiated for " + ids.size() + " products");
    }

    /**
     * Get product count.
     *
     * @return total number of products
     */
    @GetMapping("/count")
    @Timed(value = "products.count.time", description = "Time taken to count products")
    public ResponseEntity<Long> getProductCount() {
        LOGGER.info("Getting product count");
        
        long count = service.getProductCount();
        
        LOGGER.info("Total products: {}", count);
        return ResponseEntity.ok(count);
    }
}
