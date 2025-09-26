#!/bin/bash

# Create SSL directory if it doesn't exist
mkdir -p ssl

# Generate private key
openssl genrsa -out ssl/key.pem 2048

# Generate certificate signing request
openssl req -new -key ssl/key.pem -out ssl/cert.csr -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=localhost"

# Generate self-signed certificate
openssl x509 -req -days 365 -in ssl/cert.csr -signkey ssl/key.pem -out ssl/cert.pem

# Remove certificate signing request
rm ssl/cert.csr

# Set appropriate permissions
chmod 600 ssl/key.pem
chmod 644 ssl/cert.pem

echo "SSL certificates generated successfully!"
echo "Certificate: ssl/cert.pem"
echo "Private Key: ssl/key.pem"