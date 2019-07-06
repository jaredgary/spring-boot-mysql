package com.jaredgary.hn.services;

import java.util.ArrayList;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.jaredgary.hn.model.Product;
import com.jaredgary.hn.repositories.ProductRepository;

@Service
@Transactional
public class ProductService {

	@Autowired
	private ProductRepository repository;

	public List<Product> listAll() {
		final List<Product> products = new ArrayList<>();
		this.repository.findAll().forEach(products::add);
		return products;
	}

	public Product getById(Long id) {
		return this.repository.findById(id).orElse(null);
	}

	public Product saveOrUpdate(Product product) {
		this.repository.save(product);
		return product;
	}

	public void delete(Long id) {
		this.repository.deleteById(id);
	}

}
