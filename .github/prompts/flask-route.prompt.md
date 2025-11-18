# Create New Flask Route
## Task
Create a new Flask API endpoint for the YouTube transcript web application.
## Requirements
### 1. File Location
- Add to: `src/web_app.py`
- Follow existing route patterns
### 2. Basic Structure
```python
@app.route("/api/<endpoint-name>", methods=["POST"])
def endpoint_name() -> jsonify:
    """Brief description of what this endpoint does.
    Request Form Parameters:
        param1: Description (type)
        param2: Description (type, optional)
    Returns:
        JSON response with:
        - success: Boolean status
        - data: Result data (if successful)
        - error: Error message (if failed)
    HTTP Status Codes:
        200: Success
        400: Bad request (missing/invalid parameters)
        500: Server error
    """
    try:
        # Extract and validate parameters
        param1 = request.form.get('param1')
        if not param1:
            return jsonify({"success": False, "error": "Missing param1"}), 400
        logger.info(f"Processing {endpoint_name}: {param1[:50]}...")
        # Process the request
        result = process_function(param1)
        # Return success
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.exception("Unexpected error in endpoint")
        return jsonify({"success": False, "error": "Internal server error"}), 500
```
### 3. Checklist
- [ ] Added comprehensive docstring with request/response format
- [ ] Validated all required parameters
- [ ] Logged incoming requests
- [ ] Used appropriate HTTP status codes
- [ ] Handled exceptions gracefully
- [ ] Returned consistent JSON structure
- [ ] Added type hints
- [ ] Tested with various inputs
## Common Patterns
### Parameter Validation
```python
# Required parameter
video_id = request.form.get('video_id')
if not video_id:
    return jsonify({"success": False, "error": "Missing video_id"}), 400
# Optional parameter with default
translate_to = request.form.get('translate_to', '')
# Integer parameter with validation
try:
    max_sections = int(request.form.get('max_sections', 15))
except ValueError:
    return jsonify({"success": False, "error": "max_sections must be an integer"}), 400
```
### Response Formats
```python
# Success with data
return jsonify({"success": True, "data": result}), 200
# Error response
return jsonify({"success": False, "error": "Error description"}), 400
```
