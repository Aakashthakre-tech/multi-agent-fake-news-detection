from unittest.mock import MagicMock, PropertyMock

# Test 1: Does mock_response.text return the string we set via PropertyMock?
m = MagicMock()
type(m).text = PropertyMock(return_value="<html><body><p>test content here that is long enough</p></body></html>")
print(f"Type of m.text: {type(m.text)}")
print(f"m.text: {repr(m.text)[:100]}")

# Test 2: Does mock_response.text return string when set as instance attr?
m2 = MagicMock()
m2.text = "<html><body><p>direct attribute test</p></body></html>"
print(f"\nType of m2.text: {type(m2.text)}")
print(f"m2.text: {repr(m2.text)[:100]}")
