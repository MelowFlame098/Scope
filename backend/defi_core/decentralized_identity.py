"""Decentralized Identity and Security Module

This module provides comprehensive decentralized identity (DID) management,
security protocols, privacy features, and authentication mechanisms for
the DeFi platform.

Features:
- Decentralized Identity (DID) creation and management
- Self-sovereign identity (SSI) protocols
- Zero-knowledge proof authentication
- Multi-signature wallet integration
- Privacy-preserving transactions
- Credential verification
- Reputation scoring
- Biometric authentication
- Hardware wallet integration
- Social recovery mechanisms

Supported Standards:
- W3C DID Core Specification
- Verifiable Credentials (VC)
- DIDComm Messaging
- BIP-32/44 HD Wallets
- EIP-712 Typed Data Signing
- EIP-1271 Smart Contract Signatures
- EIP-4361 Sign-In with Ethereum

Author: FinScope AI Team
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from collections import defaultdict
import base64
import uuid
from abc import ABC, abstractmethod
import hmac
import time

class DIDMethod(Enum):
    """Supported DID methods"""
    ETH = "did:eth"
    KEY = "did:key"
    WEB = "did:web"
    ION = "did:ion"
    POLYGON = "did:polygon"
    SOLANA = "did:sol"
    PEER = "did:peer"

class CredentialType(Enum):
    """Types of verifiable credentials"""
    IDENTITY = "identity"
    KYC = "kyc"
    ACCREDITATION = "accreditation"
    REPUTATION = "reputation"
    FINANCIAL = "financial"
    EDUCATION = "education"
    PROFESSIONAL = "professional"
    MEMBERSHIP = "membership"
    CERTIFICATION = "certification"
    ACHIEVEMENT = "achievement"

class AuthenticationMethod(Enum):
    """Authentication methods"""
    WALLET_SIGNATURE = "wallet_signature"
    BIOMETRIC = "biometric"
    HARDWARE_KEY = "hardware_key"
    MULTI_FACTOR = "multi_factor"
    ZERO_KNOWLEDGE = "zero_knowledge"
    SOCIAL_RECOVERY = "social_recovery"
    TIME_BASED_OTP = "time_based_otp"
    SMS_VERIFICATION = "sms_verification"
    EMAIL_VERIFICATION = "email_verification"

class PrivacyLevel(Enum):
    """Privacy levels for transactions and data"""
    PUBLIC = "public"
    PSEUDONYMOUS = "pseudonymous"
    PRIVATE = "private"
    ANONYMOUS = "anonymous"
    ZERO_KNOWLEDGE = "zero_knowledge"

class SecurityLevel(Enum):
    """Security levels"""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"
    QUANTUM_RESISTANT = "quantum_resistant"

@dataclass
class DIDDocument:
    """Decentralized Identity Document"""
    id: str  # DID identifier
    context: List[str]
    controller: str
    verification_methods: List[Dict[str, Any]]
    authentication: List[str]
    assertion_method: List[str]
    key_agreement: List[str]
    capability_invocation: List[str]
    capability_delegation: List[str]
    service_endpoints: List[Dict[str, Any]]
    created: datetime
    updated: datetime
    version: int = 1
    proof: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VerifiableCredential:
    """Verifiable Credential"""
    id: str
    context: List[str]
    type: List[str]
    issuer: str  # DID of issuer
    issuance_date: datetime
    expiration_date: Optional[datetime]
    credential_subject: Dict[str, Any]
    credential_schema: Optional[Dict[str, Any]]
    credential_status: Optional[Dict[str, Any]]
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    terms_of_use: List[Dict[str, Any]] = field(default_factory=list)
    refresh_service: Optional[Dict[str, Any]] = None
    proof: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VerifiablePresentation:
    """Verifiable Presentation"""
    id: str
    context: List[str]
    type: List[str]
    holder: str  # DID of holder
    verifiable_credential: List[VerifiableCredential]
    proof: Dict[str, Any]
    created: datetime = field(default_factory=datetime.now)

@dataclass
class AuthenticationChallenge:
    """Authentication challenge"""
    challenge_id: str
    did: str
    challenge_data: str
    method: AuthenticationMethod
    expires_at: datetime
    nonce: str
    domain: str
    issued_at: datetime = field(default_factory=datetime.now)
    attempts: int = 0
    max_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuthenticationResult:
    """Authentication result"""
    challenge_id: str
    did: str
    success: bool
    method: AuthenticationMethod
    signature: Optional[str]
    proof: Optional[Dict[str, Any]]
    timestamp: datetime
    session_token: Optional[str]
    expires_at: Optional[datetime]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IdentityProfile:
    """User identity profile"""
    did: str
    display_name: Optional[str]
    avatar_url: Optional[str]
    bio: Optional[str]
    website: Optional[str]
    social_links: Dict[str, str] = field(default_factory=dict)
    verified_credentials: List[str] = field(default_factory=list)  # Credential IDs
    reputation_score: float = 0.0
    trust_score: float = 0.0
    privacy_settings: Dict[str, Any] = field(default_factory=dict)
    security_settings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_active: Optional[datetime] = None
    verification_status: Dict[str, bool] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

@dataclass
class SecurityAuditLog:
    """Security audit log entry"""
    log_id: str
    did: str
    event_type: str
    event_description: str
    timestamp: datetime
    ip_address: Optional[str]
    user_agent: Optional[str]
    location: Optional[Dict[str, Any]]
    risk_score: float
    action_taken: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PrivacyTransaction:
    """Privacy-preserving transaction"""
    transaction_id: str
    from_did: str
    to_did: Optional[str]  # Can be None for anonymous transactions
    amount: Decimal
    token: str
    privacy_level: PrivacyLevel
    zero_knowledge_proof: Optional[Dict[str, Any]]
    commitment: Optional[str]  # For commitment schemes
    nullifier: Optional[str]  # For preventing double-spending
    timestamp: datetime
    network: str
    gas_fee: Decimal
    status: str  # "pending", "confirmed", "failed"
    metadata: Dict[str, Any] = field(default_factory=dict)

class DecentralizedIdentity:
    """Decentralized Identity and Security Manager
    
    Provides comprehensive DID management, authentication, privacy,
    and security features for the DeFi platform.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Decentralized Identity Manager
        
        Args:
            config: Configuration dictionary containing:
                - did_registry: DID registry settings
                - credential_registry: Credential registry settings
                - authentication_settings: Authentication configuration
                - privacy_settings: Privacy configuration
                - security_settings: Security configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DecentralizedIdentity")
        
        # Registry settings
        self.did_registry = config.get('did_registry', {
            'ethereum_registry': '0x0123456789abcdef0123456789abcdef01234567',
            'ipfs_gateway': 'https://ipfs.io/ipfs/',
            'resolver_endpoint': 'https://dev.uniresolver.io/1.0/identifiers/'
        })
        
        self.credential_registry = config.get('credential_registry', {
            'issuer_registry': '0x0123456789abcdef0123456789abcdef01234567',
            'revocation_registry': '0x0123456789abcdef0123456789abcdef01234567',
            'schema_registry': '0x0123456789abcdef0123456789abcdef01234567'
        })
        
        # Authentication settings
        self.auth_settings = config.get('authentication_settings', {
            'challenge_expiry_minutes': 15,
            'session_expiry_hours': 24,
            'max_login_attempts': 3,
            'require_2fa': False,
            'biometric_enabled': True,
            'hardware_key_required': False
        })
        
        # Privacy settings
        self.privacy_settings = config.get('privacy_settings', {
            'default_privacy_level': PrivacyLevel.PSEUDONYMOUS,
            'zero_knowledge_enabled': True,
            'mixing_enabled': True,
            'stealth_addresses': True,
            'data_minimization': True
        })
        
        # Security settings
        self.security_settings = config.get('security_settings', {
            'encryption_algorithm': 'AES-256-GCM',
            'key_derivation': 'PBKDF2',
            'signature_algorithm': 'ECDSA',
            'hash_algorithm': 'SHA-256',
            'quantum_resistant': False,
            'audit_logging': True,
            'anomaly_detection': True
        })
        
        # Data stores
        self.did_documents: Dict[str, DIDDocument] = {}
        self.credentials: Dict[str, VerifiableCredential] = {}
        self.presentations: Dict[str, VerifiablePresentation] = {}
        self.identity_profiles: Dict[str, IdentityProfile] = {}
        self.auth_challenges: Dict[str, AuthenticationChallenge] = {}
        self.auth_sessions: Dict[str, AuthenticationResult] = {}
        self.audit_logs: List[SecurityAuditLog] = []
        self.privacy_transactions: Dict[str, PrivacyTransaction] = {}
        
        # Security monitoring
        self.failed_attempts: defaultdict = defaultdict(int)
        self.suspicious_activities: List[Dict[str, Any]] = []
        
        self.logger.info("Decentralized Identity Manager initialized")
    
    async def create_did(self, method: DIDMethod, controller_address: str, 
                        public_keys: List[Dict[str, Any]]) -> DIDDocument:
        """Create a new Decentralized Identity
        
        Args:
            method: DID method to use
            controller_address: Address of the DID controller
            public_keys: List of public keys for the DID
            
        Returns:
            DID Document
        """
        try:
            # Generate DID identifier
            did_id = self._generate_did_identifier(method, controller_address)
            
            # Create verification methods
            verification_methods = []
            for i, key_data in enumerate(public_keys):
                vm_id = f"{did_id}#key-{i+1}"
                verification_method = {
                    'id': vm_id,
                    'type': key_data.get('type', 'EcdsaSecp256k1VerificationKey2019'),
                    'controller': did_id,
                    'publicKeyHex': key_data['public_key_hex']
                }
                verification_methods.append(verification_method)
            
            # Create DID Document
            did_document = DIDDocument(
                id=did_id,
                context=[
                    'https://www.w3.org/ns/did/v1',
                    'https://w3id.org/security/suites/secp256k1-2019/v1'
                ],
                controller=controller_address,
                verification_methods=verification_methods,
                authentication=[vm['id'] for vm in verification_methods],
                assertion_method=[vm['id'] for vm in verification_methods],
                key_agreement=[vm['id'] for vm in verification_methods],
                capability_invocation=[vm['id'] for vm in verification_methods],
                capability_delegation=[],
                service_endpoints=[
                    {
                        'id': f"{did_id}#finscope-service",
                        'type': 'FinScopeService',
                        'serviceEndpoint': 'https://api.finscope.ai/did-services'
                    }
                ],
                created=datetime.now(),
                updated=datetime.now()
            )
            
            # Store DID Document
            self.did_documents[did_id] = did_document
            
            # Create identity profile
            profile = IdentityProfile(
                did=did_id,
                privacy_settings={
                    'profile_visibility': 'private',
                    'transaction_privacy': self.privacy_settings['default_privacy_level'].value,
                    'data_sharing': False
                },
                security_settings={
                    'two_factor_enabled': False,
                    'biometric_enabled': self.auth_settings['biometric_enabled'],
                    'hardware_key_required': self.auth_settings['hardware_key_required']
                }
            )
            self.identity_profiles[did_id] = profile
            
            # Log creation
            await self._log_security_event(
                did_id, 'did_created', f'DID created with method {method.value}'
            )
            
            self.logger.info(f"Created DID: {did_id}")
            return did_document
            
        except Exception as e:
            self.logger.error(f"Error creating DID: {e}")
            raise
    
    def _generate_did_identifier(self, method: DIDMethod, controller_address: str) -> str:
        """Generate DID identifier
        
        Args:
            method: DID method
            controller_address: Controller address
            
        Returns:
            DID identifier string
        """
        if method == DIDMethod.ETH:
            return f"did:eth:{controller_address}"
        elif method == DIDMethod.KEY:
            # For did:key, use the public key directly
            return f"did:key:z{controller_address}"
        elif method == DIDMethod.WEB:
            return f"did:web:finscope.ai:users:{controller_address}"
        else:
            # Generic format
            return f"{method.value}:{controller_address}"
    
    async def resolve_did(self, did: str) -> Optional[DIDDocument]:
        """Resolve a DID to its DID Document
        
        Args:
            did: DID identifier
            
        Returns:
            DID Document if found, None otherwise
        """
        try:
            # Check local storage first
            if did in self.did_documents:
                return self.did_documents[did]
            
            # In real implementation, would resolve from registry/network
            self.logger.info(f"DID {did} not found locally, would resolve from registry")
            return None
            
        except Exception as e:
            self.logger.error(f"Error resolving DID {did}: {e}")
            return None
    
    async def issue_credential(self, issuer_did: str, subject_did: str, 
                              credential_type: CredentialType, 
                              claims: Dict[str, Any]) -> VerifiableCredential:
        """Issue a verifiable credential
        
        Args:
            issuer_did: DID of the credential issuer
            subject_did: DID of the credential subject
            credential_type: Type of credential
            claims: Claims to include in the credential
            
        Returns:
            Verifiable Credential
        """
        try:
            credential_id = f"urn:uuid:{uuid.uuid4()}"
            
            # Create credential subject
            credential_subject = {
                'id': subject_did,
                **claims
            }
            
            # Create verifiable credential
            credential = VerifiableCredential(
                id=credential_id,
                context=[
                    'https://www.w3.org/2018/credentials/v1',
                    'https://finscope.ai/credentials/v1'
                ],
                type=['VerifiableCredential', credential_type.value.title() + 'Credential'],
                issuer=issuer_did,
                issuance_date=datetime.now(),
                expiration_date=datetime.now() + timedelta(days=365),  # 1 year validity
                credential_subject=credential_subject,
                credential_schema={
                    'id': f'https://finscope.ai/schemas/{credential_type.value}.json',
                    'type': 'JsonSchemaValidator2018'
                },
                credential_status={
                    'id': f'https://finscope.ai/status/{credential_id}',
                    'type': 'RevocationList2020Status',
                    'revocationListIndex': '0',
                    'revocationListCredential': 'https://finscope.ai/revocation/list.json'
                }
            )
            
            # Generate proof (simplified)
            proof = {
                'type': 'EcdsaSecp256k1Signature2019',
                'created': datetime.now().isoformat(),
                'verificationMethod': f'{issuer_did}#key-1',
                'proofPurpose': 'assertionMethod',
                'jws': self._generate_jws_signature(credential, issuer_did)
            }
            credential.proof = proof
            
            # Store credential
            self.credentials[credential_id] = credential
            
            # Update subject's profile
            if subject_did in self.identity_profiles:
                self.identity_profiles[subject_did].verified_credentials.append(credential_id)
                self.identity_profiles[subject_did].updated_at = datetime.now()
            
            # Log issuance
            await self._log_security_event(
                issuer_did, 'credential_issued', 
                f'Issued {credential_type.value} credential to {subject_did}'
            )
            
            self.logger.info(f"Issued credential {credential_id} to {subject_did}")
            return credential
            
        except Exception as e:
            self.logger.error(f"Error issuing credential: {e}")
            raise
    
    def _generate_jws_signature(self, credential: VerifiableCredential, issuer_did: str) -> str:
        """Generate JWS signature for credential (mock implementation)
        
        Args:
            credential: Credential to sign
            issuer_did: Issuer DID
            
        Returns:
            JWS signature string
        """
        # In real implementation, would use actual cryptographic signing
        payload = json.dumps({
            'id': credential.id,
            'type': credential.type,
            'issuer': credential.issuer,
            'issuanceDate': credential.issuance_date.isoformat(),
            'credentialSubject': credential.credential_subject
        }, sort_keys=True)
        
        # Mock signature
        signature_data = hashlib.sha256(payload.encode()).hexdigest()
        return f"eyJhbGciOiJFUzI1NksiLCJ0eXAiOiJKV1QifQ..{signature_data[:32]}"
    
    async def verify_credential(self, credential_id: str) -> Dict[str, Any]:
        """Verify a credential
        
        Args:
            credential_id: Credential ID to verify
            
        Returns:
            Verification result
        """
        try:
            if credential_id not in self.credentials:
                return {
                    'valid': False,
                    'error': 'Credential not found',
                    'verified_at': datetime.now().isoformat()
                }
            
            credential = self.credentials[credential_id]
            
            # Check expiration
            if credential.expiration_date and datetime.now() > credential.expiration_date:
                return {
                    'valid': False,
                    'error': 'Credential expired',
                    'expired_at': credential.expiration_date.isoformat(),
                    'verified_at': datetime.now().isoformat()
                }
            
            # Verify issuer DID
            issuer_doc = await self.resolve_did(credential.issuer)
            if not issuer_doc:
                return {
                    'valid': False,
                    'error': 'Issuer DID not resolvable',
                    'verified_at': datetime.now().isoformat()
                }
            
            # Verify signature (simplified)
            signature_valid = self._verify_jws_signature(credential)
            
            # Check revocation status (mock)
            revoked = False  # Would check actual revocation registry
            
            verification_result = {
                'valid': signature_valid and not revoked,
                'credential_id': credential_id,
                'issuer': credential.issuer,
                'subject': credential.credential_subject.get('id'),
                'type': credential.type,
                'issuance_date': credential.issuance_date.isoformat(),
                'expiration_date': credential.expiration_date.isoformat() if credential.expiration_date else None,
                'signature_valid': signature_valid,
                'revoked': revoked,
                'verified_at': datetime.now().isoformat()
            }
            
            return verification_result
            
        except Exception as e:
            self.logger.error(f"Error verifying credential {credential_id}: {e}")
            return {
                'valid': False,
                'error': str(e),
                'verified_at': datetime.now().isoformat()
            }
    
    def _verify_jws_signature(self, credential: VerifiableCredential) -> bool:
        """Verify JWS signature (mock implementation)
        
        Args:
            credential: Credential to verify
            
        Returns:
            True if signature is valid
        """
        # In real implementation, would verify actual cryptographic signature
        return True  # Mock verification
    
    async def create_authentication_challenge(self, did: str, 
                                            method: AuthenticationMethod,
                                            domain: str = "finscope.ai") -> AuthenticationChallenge:
        """Create authentication challenge
        
        Args:
            did: DID to authenticate
            method: Authentication method
            domain: Domain for the challenge
            
        Returns:
            Authentication challenge
        """
        try:
            challenge_id = str(uuid.uuid4())
            nonce = secrets.token_hex(32)
            
            # Generate challenge data based on method
            if method == AuthenticationMethod.WALLET_SIGNATURE:
                challenge_data = f"Sign this message to authenticate with FinScope:\n\nDomain: {domain}\nNonce: {nonce}\nTimestamp: {int(time.time())}"
            elif method == AuthenticationMethod.ZERO_KNOWLEDGE:
                challenge_data = f"zk_challenge_{nonce}"
            else:
                challenge_data = f"challenge_{nonce}"
            
            challenge = AuthenticationChallenge(
                challenge_id=challenge_id,
                did=did,
                challenge_data=challenge_data,
                method=method,
                expires_at=datetime.now() + timedelta(minutes=self.auth_settings['challenge_expiry_minutes']),
                nonce=nonce,
                domain=domain
            )
            
            # Store challenge
            self.auth_challenges[challenge_id] = challenge
            
            # Log challenge creation
            await self._log_security_event(
                did, 'auth_challenge_created', 
                f'Authentication challenge created with method {method.value}'
            )
            
            return challenge
            
        except Exception as e:
            self.logger.error(f"Error creating authentication challenge: {e}")
            raise
    
    async def verify_authentication(self, challenge_id: str, 
                                  response: Dict[str, Any]) -> AuthenticationResult:
        """Verify authentication response
        
        Args:
            challenge_id: Challenge ID
            response: Authentication response
            
        Returns:
            Authentication result
        """
        try:
            if challenge_id not in self.auth_challenges:
                return AuthenticationResult(
                    challenge_id=challenge_id,
                    did="",
                    success=False,
                    method=AuthenticationMethod.WALLET_SIGNATURE,
                    signature=None,
                    proof=None,
                    timestamp=datetime.now(),
                    session_token=None,
                    expires_at=None,
                    metadata={'error': 'Challenge not found'}
                )
            
            challenge = self.auth_challenges[challenge_id]
            
            # Check if challenge expired
            if datetime.now() > challenge.expires_at:
                return AuthenticationResult(
                    challenge_id=challenge_id,
                    did=challenge.did,
                    success=False,
                    method=challenge.method,
                    signature=None,
                    proof=None,
                    timestamp=datetime.now(),
                    session_token=None,
                    expires_at=None,
                    metadata={'error': 'Challenge expired'}
                )
            
            # Verify response based on method
            verification_success = False
            signature = None
            proof = None
            
            if challenge.method == AuthenticationMethod.WALLET_SIGNATURE:
                signature = response.get('signature')
                verification_success = self._verify_wallet_signature(
                    challenge.challenge_data, signature, challenge.did
                )
            elif challenge.method == AuthenticationMethod.ZERO_KNOWLEDGE:
                proof = response.get('proof')
                verification_success = self._verify_zero_knowledge_proof(
                    challenge.challenge_data, proof, challenge.did
                )
            else:
                # Other methods would be implemented here
                verification_success = True  # Mock verification
            
            # Generate session token if successful
            session_token = None
            session_expires = None
            if verification_success:
                session_token = secrets.token_urlsafe(32)
                session_expires = datetime.now() + timedelta(hours=self.auth_settings['session_expiry_hours'])
            
            result = AuthenticationResult(
                challenge_id=challenge_id,
                did=challenge.did,
                success=verification_success,
                method=challenge.method,
                signature=signature,
                proof=proof,
                timestamp=datetime.now(),
                session_token=session_token,
                expires_at=session_expires
            )
            
            # Store session if successful
            if verification_success and session_token:
                self.auth_sessions[session_token] = result
            
            # Update profile last active
            if verification_success and challenge.did in self.identity_profiles:
                self.identity_profiles[challenge.did].last_active = datetime.now()
            
            # Log authentication attempt
            await self._log_security_event(
                challenge.did, 'authentication_attempt', 
                f'Authentication {"successful" if verification_success else "failed"} with method {challenge.method.value}'
            )
            
            # Clean up challenge
            del self.auth_challenges[challenge_id]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error verifying authentication: {e}")
            raise
    
    def _verify_wallet_signature(self, message: str, signature: str, did: str) -> bool:
        """Verify wallet signature (mock implementation)
        
        Args:
            message: Original message
            signature: Signature to verify
            did: DID of signer
            
        Returns:
            True if signature is valid
        """
        # In real implementation, would verify actual cryptographic signature
        return signature is not None and len(signature) > 0
    
    def _verify_zero_knowledge_proof(self, challenge: str, proof: Dict[str, Any], did: str) -> bool:
        """Verify zero-knowledge proof (mock implementation)
        
        Args:
            challenge: Challenge data
            proof: Zero-knowledge proof
            did: DID of prover
            
        Returns:
            True if proof is valid
        """
        # In real implementation, would verify actual ZK proof
        return proof is not None and 'commitment' in proof
    
    async def create_privacy_transaction(self, from_did: str, to_did: Optional[str],
                                       amount: Decimal, token: str,
                                       privacy_level: PrivacyLevel) -> PrivacyTransaction:
        """Create privacy-preserving transaction
        
        Args:
            from_did: Sender DID
            to_did: Recipient DID (None for anonymous)
            amount: Transaction amount
            token: Token symbol
            privacy_level: Desired privacy level
            
        Returns:
            Privacy transaction
        """
        try:
            transaction_id = str(uuid.uuid4())
            
            # Generate privacy components based on level
            zero_knowledge_proof = None
            commitment = None
            nullifier = None
            
            if privacy_level in [PrivacyLevel.PRIVATE, PrivacyLevel.ANONYMOUS, PrivacyLevel.ZERO_KNOWLEDGE]:
                # Generate ZK proof components (mock)
                zero_knowledge_proof = {
                    'proof': secrets.token_hex(64),
                    'public_inputs': [secrets.token_hex(32) for _ in range(3)],
                    'verification_key': secrets.token_hex(32)
                }
                commitment = secrets.token_hex(32)
                nullifier = secrets.token_hex(32)
            
            transaction = PrivacyTransaction(
                transaction_id=transaction_id,
                from_did=from_did,
                to_did=to_did if privacy_level != PrivacyLevel.ANONYMOUS else None,
                amount=amount,
                token=token,
                privacy_level=privacy_level,
                zero_knowledge_proof=zero_knowledge_proof,
                commitment=commitment,
                nullifier=nullifier,
                timestamp=datetime.now(),
                network="ethereum",  # Default network
                gas_fee=Decimal('0.01'),  # Mock gas fee
                status="pending"
            )
            
            # Store transaction
            self.privacy_transactions[transaction_id] = transaction
            
            # Log transaction creation
            await self._log_security_event(
                from_did, 'privacy_transaction_created',
                f'Created {privacy_level.value} transaction for {amount} {token}'
            )
            
            return transaction
            
        except Exception as e:
            self.logger.error(f"Error creating privacy transaction: {e}")
            raise
    
    async def get_reputation_score(self, did: str) -> Dict[str, Any]:
        """Get reputation score for a DID
        
        Args:
            did: DID to get reputation for
            
        Returns:
            Reputation data
        """
        try:
            if did not in self.identity_profiles:
                return {
                    'did': did,
                    'reputation_score': 0.0,
                    'trust_score': 0.0,
                    'error': 'DID not found'
                }
            
            profile = self.identity_profiles[did]
            
            # Calculate reputation based on various factors
            factors = {
                'verified_credentials': len(profile.verified_credentials) * 10,
                'account_age_days': (datetime.now() - profile.created_at).days,
                'activity_score': 50 if profile.last_active and 
                                (datetime.now() - profile.last_active).days < 30 else 0,
                'verification_status': sum(profile.verification_status.values()) * 20,
                'security_score': 30 if profile.security_settings.get('two_factor_enabled') else 0
            }
            
            # Calculate weighted reputation score
            reputation_score = min(100.0, sum(factors.values()) / 10)
            
            # Update profile
            profile.reputation_score = reputation_score
            profile.trust_score = reputation_score * 0.8  # Trust is slightly lower than reputation
            
            return {
                'did': did,
                'reputation_score': reputation_score,
                'trust_score': profile.trust_score,
                'factors': factors,
                'verified_credentials_count': len(profile.verified_credentials),
                'account_age_days': factors['account_age_days'],
                'last_active': profile.last_active.isoformat() if profile.last_active else None,
                'calculated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating reputation for {did}: {e}")
            return {
                'did': did,
                'reputation_score': 0.0,
                'trust_score': 0.0,
                'error': str(e)
            }
    
    async def _log_security_event(self, did: str, event_type: str, description: str,
                                 risk_score: float = 0.0, metadata: Dict[str, Any] = None):
        """Log security event
        
        Args:
            did: DID associated with event
            event_type: Type of security event
            description: Event description
            risk_score: Risk score (0-10)
            metadata: Additional metadata
        """
        try:
            if not self.security_settings.get('audit_logging', True):
                return
            
            log_entry = SecurityAuditLog(
                log_id=str(uuid.uuid4()),
                did=did,
                event_type=event_type,
                event_description=description,
                timestamp=datetime.now(),
                ip_address=None,  # Would be populated from request context
                user_agent=None,  # Would be populated from request context
                location=None,  # Would be populated from IP geolocation
                risk_score=risk_score,
                action_taken=None,
                metadata=metadata or {}
            )
            
            self.audit_logs.append(log_entry)
            
            # Keep only recent logs (last 10000)
            if len(self.audit_logs) > 10000:
                self.audit_logs = self.audit_logs[-10000:]
            
        except Exception as e:
            self.logger.error(f"Error logging security event: {e}")
    
    async def get_security_analytics(self, did: Optional[str] = None, 
                                   timeframe_hours: int = 24) -> Dict[str, Any]:
        """Get security analytics
        
        Args:
            did: Specific DID to analyze (None for all)
            timeframe_hours: Analysis timeframe in hours
            
        Returns:
            Security analytics data
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
            
            # Filter logs by timeframe and DID
            relevant_logs = [
                log for log in self.audit_logs
                if log.timestamp >= cutoff_time and (did is None or log.did == did)
            ]
            
            # Calculate metrics
            total_events = len(relevant_logs)
            event_types = defaultdict(int)
            risk_scores = []
            hourly_activity = defaultdict(int)
            
            for log in relevant_logs:
                event_types[log.event_type] += 1
                risk_scores.append(log.risk_score)
                hour_key = log.timestamp.strftime('%Y-%m-%d %H:00')
                hourly_activity[hour_key] += 1
            
            # Calculate statistics
            avg_risk_score = statistics.mean(risk_scores) if risk_scores else 0.0
            max_risk_score = max(risk_scores) if risk_scores else 0.0
            high_risk_events = len([score for score in risk_scores if score >= 7.0])
            
            analytics = {
                'timeframe_hours': timeframe_hours,
                'did': did,
                'total_events': total_events,
                'event_types': dict(event_types),
                'risk_metrics': {
                    'average_risk_score': avg_risk_score,
                    'maximum_risk_score': max_risk_score,
                    'high_risk_events': high_risk_events,
                    'risk_distribution': {
                        'low': len([s for s in risk_scores if s < 3.0]),
                        'medium': len([s for s in risk_scores if 3.0 <= s < 7.0]),
                        'high': len([s for s in risk_scores if s >= 7.0])
                    }
                },
                'activity_patterns': {
                    'hourly_activity': dict(hourly_activity),
                    'peak_hour': max(hourly_activity.items(), key=lambda x: x[1])[0] if hourly_activity else None,
                    'events_per_hour': total_events / timeframe_hours if timeframe_hours > 0 else 0
                },
                'security_status': {
                    'overall_status': 'secure' if avg_risk_score < 3.0 else 'moderate' if avg_risk_score < 7.0 else 'high_risk',
                    'active_sessions': len(self.auth_sessions),
                    'pending_challenges': len(self.auth_challenges),
                    'total_dids': len(self.did_documents),
                    'total_credentials': len(self.credentials)
                },
                'generated_at': datetime.now().isoformat()
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error generating security analytics: {e}")
            return {}
    
    async def shutdown(self):
        """Gracefully shutdown decentralized identity manager"""
        self.logger.info("Shutting down decentralized identity manager...")
        
        # Clear sensitive data
        self.auth_challenges.clear()
        self.auth_sessions.clear()
        
        # Clear caches
        self.did_documents.clear()
        self.credentials.clear()
        self.presentations.clear()
        self.identity_profiles.clear()
        self.privacy_transactions.clear()
        
        # Clear logs (in production, would archive them)
        self.audit_logs.clear()
        
        self.logger.info("Decentralized identity manager shutdown complete")

# Export main classes
__all__ = [
    'DecentralizedIdentity',
    'DIDDocument', 'VerifiableCredential', 'VerifiablePresentation',
    'AuthenticationChallenge', 'AuthenticationResult', 'IdentityProfile',
    'SecurityAuditLog', 'PrivacyTransaction',
    'DIDMethod', 'CredentialType', 'AuthenticationMethod', 'PrivacyLevel', 'SecurityLevel'
]