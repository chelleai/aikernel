import litellm

litellm.modify_params = True  # added to support Bedrock which assumes alternating message roles, not always the case
