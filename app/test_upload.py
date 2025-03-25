import requests

# Define the API endpoint
API_URL = "http://127.0.0.1:8000/query"

# Path to the test PDF file
TEST_FILE_PATH = "test2.pdf"

# Query string to test
TEST_QUERY = "I want you to extract the first name, last name and address of the customer from the document in a json format.Also extract DL number of the customer from driver license."
def test_query():
    # Open the test file in binary mode
    with open(TEST_FILE_PATH, "rb") as file:
        # Prepare the file payload and form data
        files = {"file": (TEST_FILE_PATH, file, "application/pdf")}
        data = {"query": TEST_QUERY}
        
        # Send the POST request to the API
        response = requests.post(API_URL, files=files, data=data)
        
        # Print the response
        print("Status Code:", response.status_code)
        try:
            print("Response JSON:", response.json())
        except Exception as e:
            print("Failed to parse response as JSON:", str(e))

if __name__ == "__main__":
    test_query()