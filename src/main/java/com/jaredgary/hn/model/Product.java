package com.jaredgary.hn.model;

import java.io.Serializable;
import java.time.LocalDateTime;

import javax.persistence.*;
import javax.validation.constraints.*;

import org.hibernate.annotations.Cache;
import org.hibernate.annotations.CacheConcurrencyStrategy;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@Entity
@Table(name = "product", indexes = {
    @Index(name = "idx_product_description", columnList = "description"),
    @Index(name = "idx_product_price", columnList = "price")
})
@Cache(usage = CacheConcurrencyStrategy.READ_WRITE)
@JsonIgnoreProperties({"hibernateLazyInitializer", "handler"})
public class Product implements Serializable {

    private static final long serialVersionUID = 1L;

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private Long id;

    @NotBlank(message = "Description is required")
    @Size(min = 1, max = 255, message = "Description must be between 1 and 255 characters")
    @Column(name = "description", nullable = false, length = 255)
    private String description;

    @NotNull(message = "Price is required")
    @Positive(message = "Price must be positive")
    @DecimalMin(value = "0.01", message = "Price must be at least 0.01")
    @DecimalMax(value = "999999.99", message = "Price must not exceed 999999.99")
    @Column(name = "price", nullable = false, precision = 8, scale = 2)
    private Double price;

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    @Version
    @Column(name = "version")
    private Long version;

    public Product() {
    }

    public Product(String description, Double price) {
        this.description = description;
        this.price = price;
    }

    // Getters and Setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public Double getPrice() {
        return price;
    }

    public void setPrice(Double price) {
        this.price = price;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }

    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }

    public Long getVersion() {
        return version;
    }

    public void setVersion(Long version) {
        this.version = version;
    }

    @Override
    public String toString() {
        return "Product{" +
                "id=" + id +
                ", description='" + description + '\'' +
                ", price=" + price +
                ", createdAt=" + createdAt +
                ", updatedAt=" + updatedAt +
                ", version=" + version +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Product)) return false;
        Product product = (Product) o;
        return id != null && id.equals(product.getId());
    }

    @Override
    public int hashCode() {
        return getClass().hashCode();
    }
}
