package com.jaredgary.hn.tests;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.web.server.LocalServerPort;
import org.springframework.cache.CacheManager;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.test.context.junit4.SpringRunner;
import org.springframework.transaction.annotation.Transactional;

import com.jaredgary.hn.model.Product;
import com.jaredgary.hn.services.ProductService;

@RunWith(SpringRunner.class)
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Transactional
public class SpringBootMySqlApplicationTests {

    @LocalServerPort
    private int port;

    @Autowired
    private TestRestTemplate restTemplate;

    @Autowired
    private ProductService productService;

    @Autowired
    private CacheManager cacheManager;

    private String getRootUrl() {
        return "http://localhost:" + port + "/api/v1/products";
    }

    @Test
    public void contextLoads() {
        // Verify that the application context loads successfully
        assertNotNull(productService);
        assertNotNull(cacheManager);
    }

    @Test
    public void testCreateProduct() {
        Product product = new Product("Test Product", 99.99);
        
        ResponseEntity<Product> response = restTemplate.postForEntity(
            getRootUrl(), product, Product.class);
        
        assertEquals(HttpStatus.CREATED, response.getStatusCode());
        assertNotNull(response.getBody());
        assertNotNull(response.getBody().getId());
        assertEquals("Test Product", response.getBody().getDescription());
        assertEquals(Double.valueOf(99.99), response.getBody().getPrice());
    }

    @Test
    public void testGetProductById() {
        // First create a product
        Product product = new Product("Test Product for GET", 149.99);
        Product savedProduct = productService.saveOrUpdate(product);
        
        // Then retrieve it
        ResponseEntity<Product> response = restTemplate.getForEntity(
            getRootUrl() + "/" + savedProduct.getId(), Product.class);
        
        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertNotNull(response.getBody());
        assertEquals(savedProduct.getId(), response.getBody().getId());
        assertEquals("Test Product for GET", response.getBody().getDescription());
    }

    @Test
    public void testGetProductByIdNotFound() {
        ResponseEntity<Product> response = restTemplate.getForEntity(
            getRootUrl() + "/999999", Product.class);
        
        assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
    }

    @Test
    public void testUpdateProduct() {
        // First create a product
        Product product = new Product("Original Product", 99.99);
        Product savedProduct = productService.saveOrUpdate(product);
        
        // Update the product
        savedProduct.setDescription("Updated Product");
        savedProduct.setPrice(199.99);
        
        restTemplate.put(getRootUrl() + "/" + savedProduct.getId(), savedProduct);
        
        // Verify the update
        Product updatedProduct = productService.getById(savedProduct.getId());
        assertEquals("Updated Product", updatedProduct.getDescription());
        assertEquals(Double.valueOf(199.99), updatedProduct.getPrice());
    }

    @Test
    public void testDeleteProduct() {
        // First create a product
        Product product = new Product("Product to Delete", 99.99);
        Product savedProduct = productService.saveOrUpdate(product);
        
        // Delete the product
        restTemplate.delete(getRootUrl() + "/" + savedProduct.getId());
        
        // Verify it's deleted
        Product deletedProduct = productService.getById(savedProduct.getId());
        assertNull(deletedProduct);
    }

    @Test
    public void testSearchProducts() {
        // Create test products
        Product product1 = new Product("Laptop Computer", 999.99);
        Product product2 = new Product("Desktop Computer", 799.99);
        Product product3 = new Product("Mobile Phone", 599.99);
        
        productService.saveOrUpdate(product1);
        productService.saveOrUpdate(product2);
        productService.saveOrUpdate(product3);
        
        // Search for products containing "Computer"
        ResponseEntity<Product[]> response = restTemplate.getForEntity(
            getRootUrl() + "/search?description=Computer", Product[].class);
        
        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertNotNull(response.getBody());
        assertTrue(response.getBody().length >= 2);
    }

    @Test
    public void testPriceRangeFilter() {
        // Create test products with different prices
        Product product1 = new Product("Cheap Product", 50.0);
        Product product2 = new Product("Medium Product", 150.0);
        Product product3 = new Product("Expensive Product", 500.0);
        
        productService.saveOrUpdate(product1);
        productService.saveOrUpdate(product2);
        productService.saveOrUpdate(product3);
        
        // Find products in price range 100-200
        ResponseEntity<Product[]> response = restTemplate.getForEntity(
            getRootUrl() + "/price-range?minPrice=100&maxPrice=200", Product[].class);
        
        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertNotNull(response.getBody());
        assertTrue(response.getBody().length >= 1);
        
        // Verify all returned products are in the specified price range
        for (Product product : response.getBody()) {
            assertTrue(product.getPrice() >= 100.0 && product.getPrice() <= 200.0);
        }
    }

    @Test
    public void testBatchCreateProducts() {
        List<Product> products = Arrays.asList(
            new Product("Batch Product 1", 99.99),
            new Product("Batch Product 2", 149.99),
            new Product("Batch Product 3", 199.99)
        );
        
        ResponseEntity<Product[]> response = restTemplate.postForEntity(
            getRootUrl() + "/batch", products, Product[].class);
        
        assertEquals(HttpStatus.CREATED, response.getStatusCode());
        assertNotNull(response.getBody());
        assertEquals(3, response.getBody().length);
        
        // Verify all products have IDs assigned
        for (Product product : response.getBody()) {
            assertNotNull(product.getId());
        }
    }

    @Test
    public void testProductValidation() {
        // Test with invalid product (empty description)
        Product invalidProduct = new Product("", 99.99);
        
        ResponseEntity<String> response = restTemplate.postForEntity(
            getRootUrl(), invalidProduct, String.class);
        
        assertEquals(HttpStatus.BAD_REQUEST, response.getStatusCode());
    }

    @Test
    public void testProductValidationNegativePrice() {
        // Test with invalid product (negative price)
        Product invalidProduct = new Product("Valid Description", -10.0);
        
        ResponseEntity<String> response = restTemplate.postForEntity(
            getRootUrl(), invalidProduct, String.class);
        
        assertEquals(HttpStatus.BAD_REQUEST, response.getStatusCode());
    }

    @Test
    public void testPaginationEndpoint() {
        // Create several test products
        for (int i = 1; i <= 15; i++) {
            Product product = new Product("Product " + i, 10.0 * i);
            productService.saveOrUpdate(product);
        }
        
        // Test pagination
        ResponseEntity<String> response = restTemplate.getForEntity(
            getRootUrl() + "?page=0&size=5", String.class);
        
        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertNotNull(response.getBody());
        // The response should contain pagination metadata
        assertTrue(response.getBody().contains("totalElements"));
        assertTrue(response.getBody().contains("totalPages"));
    }

    @Test
    public void testCacheManager() {
        // Verify that cache manager is properly configured
        assertNotNull(cacheManager);
        assertNotNull(cacheManager.getCache("products"));
    }

    @Test
    public void testProductCount() {
        // Get initial count
        ResponseEntity<Long> initialResponse = restTemplate.getForEntity(
            getRootUrl() + "/count", Long.class);
        assertEquals(HttpStatus.OK, initialResponse.getStatusCode());
        Long initialCount = initialResponse.getBody();
        
        // Add a product
        Product product = new Product("Count Test Product", 99.99);
        productService.saveOrUpdate(product);
        
        // Verify count increased
        ResponseEntity<Long> finalResponse = restTemplate.getForEntity(
            getRootUrl() + "/count", Long.class);
        assertEquals(HttpStatus.OK, finalResponse.getStatusCode());
        Long finalCount = finalResponse.getBody();
        
        assertTrue(finalCount > initialCount);
    }

    @Test
    public void testPerformanceWithMultipleOperations() {
        long startTime = System.currentTimeMillis();
        
        // Perform multiple operations to test performance
        for (int i = 0; i < 10; i++) {
            Product product = new Product("Performance Test " + i, 10.0 * i);
            Product savedProduct = productService.saveOrUpdate(product);
            
            // Read the product back
            Product retrievedProduct = productService.getById(savedProduct.getId());
            assertNotNull(retrievedProduct);
            assertEquals(savedProduct.getId(), retrievedProduct.getId());
        }
        
        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;
        
        // Performance test - should complete within reasonable time (5 seconds)
        assertTrue("Performance test took too long: " + duration + "ms", duration < 5000);
    }
}
