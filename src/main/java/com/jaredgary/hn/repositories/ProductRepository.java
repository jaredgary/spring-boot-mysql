package com.jaredgary.hn.repositories;

import org.springframework.data.repository.CrudRepository;

import com.jaredgary.hn.model.Product;

public interface ProductRepository extends CrudRepository<Product, Long> {

}
