package com.jaredgary.hn.controller;

import java.util.List;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.jaredgary.hn.constants.Constants;
import com.jaredgary.hn.model.Product;
import com.jaredgary.hn.services.ProductService;

/**
 * ProductController.
 *
 * @author Gary Gonzalez Zepeda <mailto:gary.gonzalez@tigo.com.hn />
 * @version 1.0.0
 * @see
 * @since 07-15-2019 01:56:52 PM 2019
 */
@RestController
@RequestMapping("/product")
public class ProductController {

	/** Attribute that determine a Constant of LOGGER. */
	private static final Logger LOGGER = LogManager.getLogger(ProductController.class);

	/** Attribute that determine service. */
	@Autowired
	private ProductService service;

	/**
	 * List all.
	 *
	 * @return the response entity
	 */
	@PostMapping("/list")
	public ResponseEntity<List<Product>> listAll() {
		LOGGER.info("List all Products");
		final List<Product> response = this.service.listAll();
		return new ResponseEntity<List<Product>>(response, HttpStatus.OK);
	}

	/**
	 * Gets the by id.
	 *
	 * @param id the id
	 * @return the by id
	 */
	@GetMapping("/show/{id}")
	public ResponseEntity<Product> getById(@PathVariable Long id) {
		final Product response = this.service.getById(id);
		if (response == null) {
			LOGGER.info(HttpStatus.NO_CONTENT.getReasonPhrase());
			return new ResponseEntity<>(HttpStatus.NO_CONTENT);
		}
		LOGGER.info("Show Product with ID: " + id);
		return new ResponseEntity<Product>(response, HttpStatus.OK);
	}

	/**
	 * New product.
	 *
	 * @param product the product
	 * @return the response entity
	 */
	@PostMapping("/new")
	public ResponseEntity<Product> newProduct(@RequestBody Product product) {
		LOGGER.info("Add new Product");
		final Product response = this.service.saveOrUpdate(product);
		return new ResponseEntity<Product>(response, HttpStatus.OK);
	}

	/**
	 * Edits the.
	 *
	 * @param product the product
	 * @return the response entity
	 */
	@PostMapping("/edit")
	public ResponseEntity<?> edit(@RequestBody Product product) {
		final long id = product.getId();
		Product response = this.service.getById(id);
		if (response == null) {
			LOGGER.warn(HttpStatus.NOT_FOUND.getReasonPhrase());
			return new ResponseEntity<>(Constants.NOT_FOUND.getMessage(), HttpStatus.NO_CONTENT);
		}
		response = this.service.saveOrUpdate(product);
		LOGGER.info("Edit Product with ID: " + id);
		return new ResponseEntity<Product>(response, HttpStatus.OK);
	}

	/**
	 * Delete.
	 *
	 * @param id the id
	 * @return the response entity
	 */
	@GetMapping("/delete/{id}")
	public ResponseEntity<?> delete(@PathVariable Long id) {
		final Product response = this.service.getById(id);
		if (response == null) {
			LOGGER.warn(HttpStatus.NOT_FOUND.getReasonPhrase());
			return new ResponseEntity<>(Constants.NOT_FOUND.getMessage(), HttpStatus.NO_CONTENT);
		}
		this.service.delete(id);
		LOGGER.info("Delete Product with ID: " + id);
		return new ResponseEntity<>(Constants.SUCCESS.getMessage(), HttpStatus.OK);
	}

}
