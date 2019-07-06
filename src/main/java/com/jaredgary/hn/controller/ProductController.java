package com.jaredgary.hn.controller;

import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import com.jaredgary.hn.constants.Constants;
import com.jaredgary.hn.model.Product;
import com.jaredgary.hn.services.ProductService;

@RestController
@RequestMapping("/product")
public class ProductController {
	
	private static final Logger LOGGER = LogManager.getLogger(ProductController.class);

	@Autowired
	private ProductService service;

	@RequestMapping(value = "/list", method = RequestMethod.POST)
	public ResponseEntity<List<Product>> listAll() {
		LOGGER.info("List all Products");
		final List<Product> response = this.service.listAll();
		return new ResponseEntity<List<Product>>(response, HttpStatus.OK);
	}

	@RequestMapping(value = "/show/{id}", method = RequestMethod.GET)
	public ResponseEntity<Product> getById(@PathVariable Long id) {
		final Product response = this.service.getById(id);
		if (response == null) {
			LOGGER.info(HttpStatus.NO_CONTENT.getReasonPhrase());
			return new ResponseEntity<>(HttpStatus.NO_CONTENT);
		}
		LOGGER.info("Show Product with ID: " +  id);
		return new ResponseEntity<Product>(response, HttpStatus.OK);
	}

	@RequestMapping(value = "/new", method = RequestMethod.POST)
	public ResponseEntity<Product> newProduct(@RequestBody Product product) {
		LOGGER.info("Add new Product");
		final Product response = this.service.saveOrUpdate(product);
		return new ResponseEntity<Product>(response, HttpStatus.OK);
	}

	@RequestMapping(value = "/edit", method = RequestMethod.POST)
	public ResponseEntity<?> edit(@RequestBody Product product) {
		final long id = product.getId();
		Product response = this.service.getById(id);
		if (response == null) {
			LOGGER.warn(HttpStatus.NOT_FOUND.getReasonPhrase());
			return new ResponseEntity<>(Constants.NOT_FOUND.getMessage(), HttpStatus.NO_CONTENT);
		}
		response = this.service.saveOrUpdate(product);
		LOGGER.info("Edit Product with ID: " +  id);
		return new ResponseEntity<Product>(response, HttpStatus.OK);
	}

	@RequestMapping(value = "/delete/{id}", method = RequestMethod.GET)
	public ResponseEntity<?> delete(@PathVariable Long id) {
		final Product response = this.service.getById(id);
		if (response == null) {
			LOGGER.warn(HttpStatus.NOT_FOUND.getReasonPhrase());
			return new ResponseEntity<>(Constants.NOT_FOUND.getMessage(), HttpStatus.NO_CONTENT);
		}
		this.service.delete(id);
		LOGGER.info("Delete Product with ID: " +  id);
		return new ResponseEntity<>(Constants.SUCCESS.getMessage(), HttpStatus.OK);
	}

}
