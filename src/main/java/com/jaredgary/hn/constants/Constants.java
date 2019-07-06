package com.jaredgary.hn.constants;

public enum Constants {

	SUCCESS("{\"response\": \"Operation successful\"}"),

	NOT_FOUND("{\"response\": \"Product not found\"}");

	private final String message;

	private Constants(final String message) {
		this.message = message;
	}

	public String getMessage() {
		return message;
	}

}
