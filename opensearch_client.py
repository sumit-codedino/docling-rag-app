from opensearchpy import OpenSearch

auth = ('admin', '159873265Aa@')

client = OpenSearch(
    hosts=[{"host": 'search-codedinodomain-kgewyl6uvsqaf2isglpf5n2cti.aos.us-east-1.on.aws', "port": 443}],
    http_auth=auth,
    http_compress=True,  # enables gzip compression for request bodies
    use_ssl=True,
    verify_certs=True,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
)

def create_index(index_name: str, number_of_shards: int = 4):
    index_body = {
        'settings': {
            'index': {
                'number_of_shards': number_of_shards
            }
        }
    }
    response = client.indices.create(index=index_name, body=index_body)
    return response

create_index('centext_index')