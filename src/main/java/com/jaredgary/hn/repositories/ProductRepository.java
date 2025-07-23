package com.jaredgary.hn.repositories;

import java.util.List;
import java.util.Optional;

import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import com.jaredgary.hn.model.Product;

@Repository
public interface ProductRepository extends JpaRepository<Product, Long> {

    @Cacheable(value = "products", key = "#id")
    @Override
    Optional<Product> findById(Long id);

    @Cacheable(value = "products", key = "'all'")
    @Override
    List<Product> findAll();

    @Cacheable(value = "products", key = "'page_' + #pageable.pageNumber + '_' + #pageable.pageSize")
    Page<Product> findAll(Pageable pageable);

    @Query("SELECT p FROM Product p WHERE p.description LIKE %:description%")
    @Cacheable(value = "products", key = "'search_' + #description")
    List<Product> findByDescriptionContaining(@Param("description") String description);

    @Query("SELECT p FROM Product p WHERE p.price BETWEEN :minPrice AND :maxPrice")
    @Cacheable(value = "products", key = "'price_range_' + #minPrice + '_' + #maxPrice")
    List<Product> findByPriceBetween(@Param("minPrice") Double minPrice, @Param("maxPrice") Double maxPrice);

    @Query("SELECT p FROM Product p WHERE p.description LIKE %:description% AND p.price BETWEEN :minPrice AND :maxPrice")
    List<Product> findByDescriptionContainingAndPriceBetween(
            @Param("description") String description,
            @Param("minPrice") Double minPrice,
            @Param("maxPrice") Double maxPrice);

    @Query("SELECT COUNT(p) FROM Product p")
    @Cacheable(value = "products", key = "'count'")
    long countProducts();

    @Modifying
    @CacheEvict(value = "products", allEntries = true)
    @Query("UPDATE Product p SET p.price = p.price * :multiplier WHERE p.id IN :ids")
    void updatePricesByIds(@Param("ids") List<Long> ids, @Param("multiplier") Double multiplier);

    @CacheEvict(value = "products", allEntries = true)
    @Override
    void deleteById(Long id);

    @CacheEvict(value = "products", allEntries = true)
    @Override
    <S extends Product> S save(S entity);
}
