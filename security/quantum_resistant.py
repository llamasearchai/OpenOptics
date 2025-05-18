import os
import base64
import json
import time
import uuid
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import datetime
import hashlib
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.exceptions import InvalidSignature, InvalidTag

# Try to import liboqs for post-quantum cryptography
try:
    import oqs
    LIBOQS_AVAILABLE = True
except ImportError:
    LIBOQS_AVAILABLE = False
    logging.warning("liboqs not available. Falling back to classical crypto primitives.")

logger = logging.getLogger(__name__)

class QuantumResistantSecurity:
    """Quantum-resistant cryptographic security manager for optical networks"""
    
    def __init__(self):
        """Initialize the quantum-resistant security manager"""
        self.key_pairs: Dict[str, Dict[str, Any]] = {}
        self.shared_secrets_store: Dict[str, Dict[str, Any]] = {} # For persistently stored shared secrets if any
        self.secure_channel_store: Dict[str, Dict[str, Any]] = {}
        self.key_exchange_sessions: Dict[str, Dict[str, Any]] = {} # For active KEM sessions

        # Configuration - will be updated by initialize()
        self.config: Dict[str, Any] = {
            "enabled": False,
            "use_quantum_random": False,
            "algorithms": {
                "key_exchange": "Kyber768", # Default KEM
                "digital_signature": "Dilithium3", # Default Signature
                "symmetric_encryption": "ChaCha20Poly1305" # Default Symmetric
            }
        }
        self.enabled_algorithms: Dict[str, Any] = {}
        self.quantum_random_source: Optional[Dict[str, Any]] = None
        
        # Supported algorithms
        self.supported_kem_algs: List[str] = []
        self.supported_sig_algs: List[str] = []
        
        if LIBOQS_AVAILABLE:
            # Initialize supported PQ algorithms
            self.supported_kem_algs = [
                "Kyber512", "Kyber768", "Kyber1024",
                "NTRU-HPS-2048-509", "NTRU-HPS-2048-677",
                "SABER-LIGHTSABER", "SABER-SABER", "SABER-FIRESABER"
            ]
            
            self.supported_sig_algs = [
                "Dilithium2", "Dilithium3", "Dilithium5",
                "Falcon-512", "Falcon-1024",
                "SPHINCS+-SHA256-128s-simple", "SPHINCS+-SHA256-192s-simple"
            ]
            
            # Filter to only include algorithms actually supported by the library
            self.supported_kem_algs = [alg for alg in self.supported_kem_algs 
                                       if alg in oqs.get_enabled_KEM_mechanisms()]
            self.supported_sig_algs = [alg for alg in self.supported_sig_algs 
                                       if alg in oqs.get_enabled_sig_mechanisms()]
            
            logger.info(f"Initialized quantum-resistant security with {len(self.supported_kem_algs)} KEM and "
                        f"{len(self.supported_sig_algs)} signature algorithms.")
        else:
            # Fallback when liboqs is not available
            logger.warning("Using classical cryptography fallbacks (ChaCha20-Poly1305, Ed25519).")
        
        # Default algorithms from available ones, or hardcoded if liboqs not present
        self.default_kem_alg = self.supported_kem_algs[0] if self.supported_kem_algs else self.config["algorithms"]["key_exchange"]
        self.default_sig_alg = self.supported_sig_algs[0] if self.supported_sig_algs else self.config["algorithms"]["digital_signature"]

    async def initialize(self, config_update: Optional[Dict[str, Any]] = None):
        """Initialize quantum-resistant security features"""
        if config_update:
            # Simple way to update nested dicts; a more robust deep merge might be needed for complex configs
            for key, value in config_update.items():
                if key in self.config and isinstance(self.config[key], dict) and isinstance(value, dict):
                    self.config[key].update(value)
                else:
                    self.config[key] = value
        
        cfg = self.config # Use the class member self.config
        logger.info(f"Initializing QuantumResistantSecurity with config: {cfg}")

        if cfg.get("enabled", False): # Safely get "enabled"
            await self._initialize_algorithms(cfg.get("algorithms", {}))
            if cfg.get("use_quantum_random", False):
                await self._initialize_quantum_random()
        
        return {
            "status": "initialized",
            "quantum_resistant_enabled": cfg.get("enabled", False),
            "algorithms": cfg.get("algorithms", {}),
            "quantum_random_available": self.quantum_random_source is not None
        }

    async def _initialize_algorithms(self, algorithm_config: Dict[str, str]):
        logger.info(f"Initializing algorithms with config: {algorithm_config}")
        # Mock initializations
        self.enabled_algorithms["key_exchange"] = {"name": algorithm_config.get("key_exchange"), "status": "enabled", "type": "mock_pqc"}
        self.enabled_algorithms["digital_signature"] = {"name": algorithm_config.get("digital_signature"), "status": "enabled", "type": "mock_pqc"}
        self.enabled_algorithms["symmetric_encryption"] = {"name": algorithm_config.get("symmetric_encryption"), "status": "enabled", "type": "mock_classical"}
        logger.info(f"Mock algorithms initialized: {self.enabled_algorithms}")

    async def _initialize_quantum_random(self):
        if not aiohttp:
            logger.warning("aiohttp is not installed. Cannot initialize ANU QRNG. Quantum random source disabled.")
            self.quantum_random_source = None
            return

        logger.info("Attempting to initialize quantum random source (ANU QRNG)...")
        # For placeholder, assume it connects if aiohttp is present
        self.quantum_random_source = {
            "type": "api", "name": "ANU Quantum Random Numbers", 
            "url": "https://qrng.anu.edu.au/API/jsonI.php", "status": "mock_connected"
        }
        # Test call could be made here in a real scenario
        logger.info("Mock Quantum random source initialized.")


    async def _get_random_bytes(self, num_bytes: int) -> bytes:
        # Simplified: uses os.urandom. Real QRNG logic was complex and external.
        logger.info(f"Generating {num_bytes} random bytes using os.urandom.")
        return os.urandom(num_bytes)

    async def get_security_status(self) -> Dict[str, Any]:
        return {
            "status": "success",
            "quantum_resistant_enabled": self.config.get("enabled", False),
            "configured_algorithms": self.config.get("algorithms"),
            "active_algorithms_summary": self.enabled_algorithms,
            "key_exchange_sessions_active": len(self.key_exchange_sessions),
            "stored_keys_count": len(self.key_pairs),
            "quantum_random_status": self.quantum_random_source["status"] if self.quantum_random_source else "unavailable"
        }

    async def generate_key_pair(self, algorithm: Optional[str] = "default") -> Dict[str, Any]:
        algo_to_use = self.config.get("algorithms", {}).get("key_exchange", self.default_kem_alg) if algorithm == "default" else algorithm
        
        if not algo_to_use:
            logger.error("No key exchange algorithm specified or configured.")
            return {"status": "error", "message": "No key exchange algorithm specified."}

        key_id = str(uuid.uuid4())
        public_key_b64: Optional[str] = None
        private_key_b64: Optional[str] = None # For mock, store b64 directly
        raw_public_key: Optional[bytes] = None
        raw_private_key: Optional[bytes] = None

        logger.info(f"Generating key pair for algorithm: {algo_to_use}")

        if LIBOQS_AVAILABLE and algo_to_use in self.supported_kem_algs:
            try:
                with oqs.KeyEncapsulation(algo_to_use) as kem:
                    raw_public_key = kem.generate_keypair()
                    raw_private_key = kem.export_secret_key()
                    public_key_b64 = base64.b64encode(raw_public_key).decode('ascii')
                    # Private key bytes are stored raw, not b64, for direct use with oqs
                    logger.info(f"Successfully generated OQS KEM key pair for {algo_to_use}.")
            except oqs.MechanismNotSupportedError:
                logger.warning(f"OQS KEM algorithm {algo_to_use} not supported by liboqs, falling back to mock.")
            except Exception as e:
                logger.error(f"Error generating OQS KEM key pair for {algo_to_use}: {e}")
                return {"status": "error", "message": f"OQS KEM key generation failed: {e}"}
        
        # Fallback to mock or if OQS failed above for KEM, or if specified algo is not an OQS KEM
        if not public_key_b64: # If OQS generation didn't happen or failed
            logger.info(f"Using mock key generation for algorithm: {algo_to_use}")
            public_key_mock_bytes = f"mock_pk_{key_id}_{algo_to_use}".encode()
            private_key_mock_bytes = f"mock_sk_{key_id}_{algo_to_use}".encode()
            public_key_b64 = base64.b64encode(public_key_mock_bytes).decode('ascii')
            private_key_b64 = base64.b64encode(private_key_mock_bytes).decode('ascii')
            raw_public_key = public_key_mock_bytes
            raw_private_key = private_key_mock_bytes


        self.key_pairs[key_id] = {
            "public_key_b64": public_key_b64,
            "private_key_b64": private_key_b64, # Storing b64 for mock, raw for OQS sk might be better
            "raw_public_key": raw_public_key,
            "raw_private_key": raw_private_key, # Store raw private key for OQS
            "algorithm": algo_to_use,
            "type": "kem_pair", # More specific type
            "created_at": time.time()
        }
        return {
            "status": "success",
            "key_id": key_id,
            "algorithm": algo_to_use,
            "public_key_b64": public_key_b64 
        }

    async def initiate_key_exchange(self, peer_id: str, algorithm: Optional[str] = "default") -> Dict[str, Any]:
        algo_to_use = self.config.get("algorithms", {}).get("key_exchange", self.default_kem_alg) if algorithm == "default" else algorithm
        if not algo_to_use:
            logger.error("No key exchange algorithm specified or configured for KEM initiation.")
            return {"status": "error", "message": "No key exchange algorithm specified for KEM initiation."}

        session_id = str(uuid.uuid4())
        
        # Generate an ephemeral key pair for this session
        key_gen_result = await self.generate_key_pair(algorithm=algo_to_use)
        if key_gen_result["status"] != "success":
            return key_gen_result
        
        session_key_id = key_gen_result["key_id"]
        our_public_key = key_gen_result["public_key_b64"]

        self.key_exchange_sessions[session_id] = {
            "id": session_id, "peer_id": peer_id, "algorithm": algo_to_use,
            "status": "initiated", "our_key_id": session_key_id,
            "our_public_key_b64": our_public_key, # Store our public key
            "created_at": datetime.datetime.now().isoformat()
        }
        logger.info(f"Initiated mock KEM session {session_id} with peer {peer_id} using {algo_to_use}. Our PK: {our_public_key[:20]}...")
        return {
            "status": "success", 
            "session_id": session_id,
            "public_key_to_send_to_peer": our_public_key, # This is our KEM public key for peer to encapsulate against
            "algorithm": algo_to_use
        }

    async def complete_key_exchange(self, session_id: str, peer_public_key_or_ciphertext_b64: str) -> Dict[str, Any]:
        session = self.key_exchange_sessions.get(session_id)
        if not session:
            return {"status": "error", "message": "Key exchange session not found."}
        if session["status"] != "initiated":
            return {"status": "error", "message": f"Session {session_id} not in 'initiated' state."}

        # In a real KEM:
        # If we initiated (sent our PK), peer_public_key_or_ciphertext_b64 is the KEM ciphertext from peer.
        # We use our SK to decapsulate it to get the shared secret.
        
        # Mock decapsulation: shared secret is derived from session details + peer's contribution
        mock_shared_secret_str = f"shared_secret_for_{session_id}_with_peer_data_{peer_public_key_or_ciphertext_b64[:10]}"
        shared_secret_bytes = hashlib.sha256(mock_shared_secret_str.encode()).digest() # Use a hash for some byte diversity
        
        shared_secret_key_id = f"ssk_{session_id}"
        self.symmetric_keys_store[shared_secret_key_id] = shared_secret_bytes
        
        session["status"] = "completed"
        session["shared_secret_key_id"] = shared_secret_key_id
        session["completed_at"] = datetime.datetime.now().isoformat()
        
        logger.info(f"Mock KEM session {session_id} completed. Shared secret ID: {shared_secret_key_id}")
        return {"status": "success", "session_id": session_id, "shared_secret_key_id": shared_secret_key_id}

    async def encrypt_data(self, data_base64: str, key_id: Optional[str] = None) -> Dict[str, Any]:
        if key_id is None or key_id not in self.symmetric_keys_store:
            # Fallback to a generic mock app key if specific session key not found/provided
            key_id = "mock_generic_app_key"
            if key_id not in self.symmetric_keys_store:
                 self.symmetric_keys_store[key_id] = os.urandom(32) # AES-256 needs 32 bytes
            logger.warning(f"Encrypting with generic app key {key_id} as specific key_id was not provided or found.")
        
        key = self.symmetric_keys_store[key_id]
        nonce = os.urandom(12) # GCM standard nonce size
        
        try:
            data_to_encrypt = base64.b64decode(data_base64)
        except Exception as e:
            return {"status": "error", "message": f"Invalid base64 data for encryption: {e}"}

        # Mock encryption (e.g., XOR with key - NOT SECURE, FOR DEMO ONLY)
        # A real implementation would use AES-GCM from 'cryptography' library
        # For simplicity, just prefixing and suffixing.
        # ciphertext_bytes = bytes([db ^ kb for db, kb in zip(data_to_encrypt, itertools.cycle(key))]) # Example XOR
        ciphertext_bytes = b"mock_encrypted_prefix_" + data_to_encrypt + b"_suffix_" + nonce
        
        logger.info(f"Mock encryption performed with key_id {key_id}.")
        return {
            "status": "success",
            "ciphertext_base64": base64.b64encode(ciphertext_bytes).decode('ascii'),
            "nonce_base64": base64.b64encode(nonce).decode('ascii'), # Nonce is crucial for GCM
            "key_id_used": key_id
        }

    async def decrypt_data(self, ciphertext_base64: str, nonce_base64: str, key_id: str) -> Dict[str, Any]:
        if key_id not in self.symmetric_keys_store:
            return {"status": "error", "message": f"Key ID {key_id} not found for decryption."}
        
        key = self.symmetric_keys_store[key_id]
        try:
            ciphertext_bytes = base64.b64decode(ciphertext_base64)
            nonce_bytes = base64.b64decode(nonce_base64) # Nonce would be used by real AES-GCM
        except Exception as e:
            return {"status": "error", "message": f"Invalid base64 data for decryption: {e}"}

        # Mock decryption (must match mock encryption)
        prefix = b"mock_encrypted_prefix_"
        suffix = b"_suffix_" + nonce_bytes # Nonce used in mock suffix to make it unique
        
        if ciphertext_bytes.startswith(prefix) and ciphertext_bytes.endswith(suffix):
            plaintext_bytes = ciphertext_bytes[len(prefix):-len(suffix)]
        else:
            logger.error(f"Mock decryption failed for key_id {key_id} due to format mismatch.")
            return {"status": "error", "message": "Mock decryption failed (format mismatch or wrong key)."}

        logger.info(f"Mock decryption performed for key_id {key_id}.")
        return {
            "status": "success",
            "plaintext": plaintext_bytes, # Return raw bytes, route will base64 encode if needed
            "key_id_used": key_id
        }

    async def sign_data(self, data_base64: str, key_id: Optional[str] = None) -> Dict[str, Any]:
        # Generate a new signature key pair with a signature algorithm
        sig_alg = self.default_sig_alg
        
        try:
            with oqs.Signature(sig_alg) as signer:
                public_key = signer.generate_keypair()
                private_key = signer.export_secret_key()
                
                key_id = str(uuid.uuid4())
                
                # Store signature key pair
                self.key_pairs[key_id] = {
                    "private_key": private_key,
                    "public_key": public_key,
                    "algorithm": sig_alg,
                    "type": "signature",
                    "created_at": time.time()
                }
        except Exception as e:
            logger.error(f"Error generating signature key pair: {e}")
            return {
                "status": "error",
                "message": f"Failed to generate signature key pair: {str(e)}"
            }
    
        # Check if key exists
        if key_id not in self.key_pairs:
            return {
                "status": "error",
                "message": f"Key ID {key_id} not found."
            }
        
        key_pair = self.key_pairs[key_id]
        algorithm = key_pair["algorithm"]
        
        # Sign data using quantum-resistant signature
        with oqs.Signature(algorithm) as signer:
            signer.load_secret_key(key_pair["private_key"])
            signature = signer.sign(data)
            
            return {
                "status": "success",
                "key_id": key_id,
                "algorithm": algorithm,
                "signature_b64": base64.b64encode(signature).decode('ascii')
            }

    async def verify_signature(self, data_base64: str, signature: str, key_id: str) -> Dict[str, Any]:
        """Verify a quantum-resistant digital signature"""
        try:
            # Decode data and signature
            data = base64.b64decode(data_base64)
            signature_bytes = base64.b64decode(signature)
            
            # Check if key exists
            if key_id not in self.key_pairs:
                return {
                    "status": "error",
                    "message": f"Key ID {key_id} not found."
                }
            
            key_pair = self.key_pairs[key_id]
            algorithm = key_pair["algorithm"]
            
            if not LIBOQS_AVAILABLE or algorithm == "HMAC-SHA256":
                # Verify using classical HMAC
                logger.info("Using classical crypto for signature verification.")
                
                try:
                    h = hmac.HMAC(key_pair["private_key"], hashes.SHA256())
                    h.update(data)
                    h.verify(signature_bytes)
                    verified = True
                except InvalidSignature:
                    verified = False
            else:
                # Verify using quantum-resistant signature
                with oqs.Signature(algorithm) as verifier:
                    verified = verifier.verify(data, signature_bytes, key_pair["public_key"])
            
            return {
                "status": "success",
                "verified": verified,
                "algorithm": algorithm
            }
        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            return {
                "status": "error",
                "message": f"Failed to verify signature: {str(e)}"
            }

    async def establish_secure_channel(self, device_id: str, peer_public_key_or_ct_b64: Optional[str] = None) -> Dict[str, Any]:
        """Establish a quantum-resistant secure channel with a device
        
        This function can be called in two ways:
        1. Initiator: Without peer_public_key_or_ct_b64 - generates keypair and returns the public key
        2. Responder: With peer_public_key_or_ct_b64 - uses the initiator's public key to complete the exchange
        """
        channel_id = f"channel_{device_id}_{str(uuid.uuid4())[:8]}"
        
        # Check if we're the initiator or responder
        if not peer_public_key_or_ct_b64:
            # We're the initiator - generate a key pair and return the public key
            key_pair_result = await self.generate_key_pair()
            
            if key_pair_result["status"] != "success":
                return key_pair_result
            
            key_id = key_pair_result["key_id"]
            public_key_b64 = key_pair_result["public_key_b64"]
            algorithm = key_pair_result["algorithm"]
            
            # Store channel info
            self.secure_channels[channel_id] = {
                "device_id": device_id,
                "key_id": key_id,
                "state": "initiated",
                "algorithm": algorithm,
                "created_at": time.time(),
                "last_activity": time.time()
            }
            
            return {
                "status": "success",
                "channel_id": channel_id,
                "public_key_b64": public_key_b64,
                "algorithm": algorithm,
                "state": "initiated"
            }
        else:
            # We're the responder - process the initiator's public key
            try:
                # Decode peer's data
                peer_data = base64.b64decode(peer_public_key_or_ct_b64)
                
                if not LIBOQS_AVAILABLE:
                    # Fallback to classical key derivation
                    logger.warning("Using classical crypto fallback for secure channel.")
                    
                    # Generate our key pair
                    key_pair_result = await self.generate_key_pair()
                    if key_pair_result["status"] != "success":
                        return key_pair_result
                    
                    key_id = key_pair_result["key_id"]
                    public_key = base64.b64decode(key_pair_result["public_key_b64"])
                    
                    # Derive shared secret using PBKDF2
                    kdf = PBKDF2HMAC(
                        algorithm=hashes.SHA256(),
                        length=32,
                        salt=peer_data,
                        iterations=100000
                    )
                    shared_secret = kdf.derive(self.key_pairs[key_id]["private_key"])
                    
                    # Store the channel with shared secret
                    self.secure_channels[channel_id] = {
                        "device_id": device_id,
                        "key_id": key_id,
                        "shared_secret": shared_secret,
                        "state": "established",
                        "algorithm": "PBKDF2-SHA256-Fallback",
                        "created_at": time.time(),
                        "last_activity": time.time()
                    }
                    
                    return {
                        "status": "success",
                        "channel_id": channel_id,
                        "state": "established",
                        "public_key_b64": key_pair_result["public_key_b64"],
                        "algorithm": "PBKDF2-SHA256-Fallback",
                        "warning": "Using classical cryptography fallback."
                    }
                
                # Try to determine if this is a public key or a ciphertext from KEM
                # based on our supported algorithms and expected sizes
                # This is a simplification; real code should have a more robust protocol
                
                # First, assume it's a public key for KEM
                for alg in self.supported_kem_algs:
                    try:
                        with oqs.KeyEncapsulation(alg) as kem:
                            # Generate our keypair
                            secret_key = kem.generate_keypair()
                            public_key = kem.export_public_key()
                            
                            # Try to encapsulate with the peer's public key
                            try:
                                shared_secret, ciphertext = kem.encap_secret(peer_data)
                                
                                # It worked, store our keys and the shared secret
                                key_id = str(uuid.uuid4())
                                self.key_pairs[key_id] = {
                                    "private_key": kem.export_secret_key(),
                                    "public_key": public_key,
                                    "algorithm": alg,
                                    "created_at": time.time()
                                }
                                
                                # Store the channel with shared secret
                                self.secure_channels[channel_id] = {
                                    "device_id": device_id,
                                    "key_id": key_id,
                                    "shared_secret": shared_secret,
                                    "state": "established",
                                    "algorithm": alg,
                                    "created_at": time.time(),
                                    "last_activity": time.time()
                                }
                                
                                return {
                                    "status": "success",
                                    "channel_id": channel_id,
                                    "state": "established",
                                    "ciphertext_b64": base64.b64encode(ciphertext).decode('ascii'),
                                    "algorithm": alg
                                }
                            except Exception:
                                # Not the right algorithm, try the next one
                                continue
                    except Exception as e:
                        logger.debug(f"Error trying KEM algorithm {alg}: {e}")
                
                # If we get here, it wasn't a valid public key for any of our KEM algorithms
                # Try to treat it as a ciphertext from KEM decapsulation
                
                # We need to have an initiated channel for this to make sense
                # Find any channel for this device that's in the initiated state
                device_channels = [c for c_id, c in self.secure_channels.items() 
                                  if c["device_id"] == device_id and c["state"] == "initiated"]
                
                if not device_channels:
                    # No initiated channel, generate a new key pair
                    key_pair_result = await self.generate_key_pair()
                    if key_pair_result["status"] != "success":
                        return key_pair_result
                    
                    key_id = key_pair_result["key_id"]
                    algorithm = key_pair_result["algorithm"]
                    
                    # Store channel info
                    self.secure_channels[channel_id] = {
                        "device_id": device_id,
                        "key_id": key_id,
                        "state": "initiated",
                        "algorithm": algorithm,
                        "created_at": time.time(),
                        "last_activity": time.time()
                    }
                    
                    return {
                        "status": "success",
                        "channel_id": channel_id,
                        "public_key_b64": key_pair_result["public_key_b64"],
                        "algorithm": algorithm,
                        "state": "initiated",
                        "message": "Couldn't process peer data, re-initiating channel."
                    }
                
                # Try to complete the key exchange with the ciphertext
                channel = device_channels[0]
                channel_id = next(c_id for c_id, c in self.secure_channels.items() if c == channel)
                key_id = channel["key_id"]
                algorithm = channel["algorithm"]
                
                try:
                    with oqs.KeyEncapsulation(algorithm) as kem:
                        kem.load_secret_key(self.key_pairs[key_id]["private_key"])
                        shared_secret = kem.decap_secret(peer_data)
                        
                        # Update channel with shared secret
                        channel["shared_secret"] = shared_secret
                        channel["state"] = "established"
                        channel["last_activity"] = time.time()
                        
                        return {
                            "status": "success",
                            "channel_id": channel_id,
                            "state": "established",
                            "algorithm": algorithm
                        }
                except Exception as e:
                    logger.error(f"Error completing secure channel: {e}")
                    return {
                        "status": "error",
                        "message": f"Failed to establish secure channel: {str(e)}"
                    }
            except Exception as e:
                logger.error(f"Error establishing secure channel: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to establish secure channel: {str(e)}"
                }                
    async def secure_channel_send(self, channel_id: str, data: bytes) -> Dict[str, Any]:
        """Send data over an established secure channel."""
        if channel_id not in self.secure_channels or self.secure_channels[channel_id]["state"] != "established":
            return {"status": "error", "message": f"Secure channel {channel_id} not established or not found."}
        
        channel = self.secure_channels[channel_id]
        shared_secret = channel.get("shared_secret")
        if not shared_secret:
            return {"status": "error", "message": f"No shared secret found for channel {channel_id}."}

        try:
            aead_cipher = ChaCha20Poly1305(shared_secret)
            nonce = os.urandom(12) # ChaCha20Poly1305 uses a 12-byte nonce
            ciphertext = aead_cipher.encrypt(nonce, data, None) # No associated data
            
            channel["last_activity"] = time.time()
            
            return {
                "status": "success",
                "channel_id": channel_id,
                "ciphertext_b64": base64.b64encode(ciphertext).decode('ascii'),
                "nonce_b64": base64.b64encode(nonce).decode('ascii')
            }
        except Exception as e:
            logger.error(f"Error sending secure data on channel {channel_id}: {e}")
            return {"status": "error", "message": f"Failed to send secure data: {str(e)}"}

    async def secure_channel_receive(self, channel_id: str, ciphertext_b64: str, nonce_b64: str) -> Dict[str, Any]:
        """Receive and decrypt data from an established secure channel."""
        if channel_id not in self.secure_channels or self.secure_channels[channel_id]["state"] != "established":
            return {"status": "error", "message": f"Secure channel {channel_id} not established or not found."}
        
        channel = self.secure_channels[channel_id]
        shared_secret = channel.get("shared_secret")
        if not shared_secret:
            return {"status": "error", "message": f"No shared secret found for channel {channel_id}."}

        try:
            ciphertext_bytes = base64.b64decode(ciphertext_b64)
            nonce_bytes = base64.b64decode(nonce_b64)
            
            aead_cipher = ChaCha20Poly1305(shared_secret)
            plaintext = aead_cipher.decrypt(nonce_bytes, ciphertext_bytes, None) # No associated data
            
            channel["last_activity"] = time.time()
            
            return {
                "status": "success",
                "channel_id": channel_id,
                "plaintext": plaintext # Return raw bytes
            }
        except InvalidTag:
            logger.error(f"Decryption failed for channel {channel_id}: authentication tag verification failed.")
            return {"status": "error", "message": "Decryption failed: authentication tag verification failed."}
        except Exception as e:
            logger.error(f"Error receiving secure data on channel {channel_id}: {e}")
            return {"status": "error", "message": f"Failed to receive secure data: {str(e)}"}
    
    async def close_secure_channel(self, channel_id: str) -> Dict[str, Any]:
        """Close a secure channel"""
        if channel_id not in self.secure_channels:
            return {
                "status": "error",
                "message": f"Channel {channel_id} not found."
            }
        
        try:
            # Get channel info before removal
            channel = self.secure_channels[channel_id]
            device_id = channel["device_id"]
            
            # Remove the channel
            del self.secure_channels[channel_id]
            
            return {
                "status": "success",
                "channel_id": channel_id,
                "device_id": device_id,
                "message": "Secure channel closed successfully."
            }
        except Exception as e:
            logger.error(f"Error closing secure channel: {e}")
            return {
                "status": "error",
                "message": f"Failed to close secure channel: {str(e)}"
            }
    
    # Key management utility methods
    
    async def list_keys(self) -> Dict[str, Any]:
        """List all keys in the manager"""
        key_info = []
        
        for key_id, key_data in self.key_pairs.items():
            key_info.append({
                "key_id": key_id,
                "algorithm": key_data["algorithm"],
                "type": key_data.get("type", "kem"),  # Default to "kem" for backward compatibility
                "created_at": key_data["created_at"],
                "has_private_key": bool(key_data.get("private_key")),
                "has_public_key": bool(key_data.get("public_key"))
            })
        
        return {
            "status": "success",
            "keys": key_info
        }
    
    async def list_shared_secrets(self) -> Dict[str, Any]:
        """List all shared secrets in the manager"""
        secret_info = []
        
        for secret_id, secret_data in self.shared_secrets_store.items():
            secret_info.append({
                "secret_id": secret_id,
                "algorithm": secret_data["algorithm"],
                "peer_id": secret_data["peer_id"],
                "created_at": secret_data["created_at"]
            })
        
        return {
            "status": "success",
            "shared_secrets": secret_info
        }
    
    async def list_secure_channels(self) -> Dict[str, Any]:
        """List all secure channels in the manager"""
        channel_info = []
        
        for channel_id, channel_data in self.secure_channels.items():
            channel_info.append({
                "channel_id": channel_id,
                "device_id": channel_data["device_id"],
                "state": channel_data["state"],
                "algorithm": channel_data["algorithm"],
                "created_at": channel_data["created_at"],
                "last_activity": channel_data["last_activity"]
            })
        
        return {
            "status": "success",
            "channels": channel_info
        }
    
    async def delete_key(self, key_id: str) -> Dict[str, Any]:
        """Delete a key pair"""
        if key_id not in self.key_pairs:
            return {
                "status": "error",
                "message": f"Key ID {key_id} not found."
            }
        
        try:
            # Check if key is being used by any secure channels
            channels_using_key = [c_id for c_id, c in self.secure_channels.items() if c.get("key_id") == key_id]
            
            if channels_using_key:
                return {
                    "status": "error",
                    "message": f"Cannot delete key: it is being used by {len(channels_using_key)} secure channels."
                }
            
            # Remove the key
            del self.key_pairs[key_id]
            
            return {
                "status": "success",
                "message": f"Key {key_id} deleted successfully."
            }
        except Exception as e:
            logger.error(f"Error deleting key: {e}")
            return {
                "status": "error",
                "message": f"Failed to delete key: {str(e)}"
            }
    
    async def delete_shared_secret(self, secret_id: str) -> Dict[str, Any]:
        """Delete a shared secret"""
        if secret_id not in self.shared_secrets_store:
            return {
                "status": "error",
                "message": f"Shared secret ID {secret_id} not found."
            }
        
        try:
            # Check if secret is being used by any secure channels
            channels_using_secret = [c_id for c_id, c in self.secure_channels.items() 
                                    if "shared_secret" in c and c["shared_secret"] == self.shared_secrets_store[secret_id]["secret"]]
            
            if channels_using_secret:
                return {
                    "status": "error",
                    "message": f"Cannot delete shared secret: it is being used by {len(channels_using_secret)} secure channels."
                }
            
            # Remove the shared secret
            del self.shared_secrets_store[secret_id]
            
            return {
                "status": "success",
                "message": f"Shared secret {secret_id} deleted successfully."
            }
        except Exception as e:
            logger.error(f"Error deleting shared secret: {e}")
            return {
                "status": "error",
                "message": f"Failed to delete shared secret: {str(e)}"
            }
    
    async def rotate_channel_keys(self, channel_id: str) -> Dict[str, Any]:
        """Rotate keys for a secure channel"""
        if channel_id not in self.secure_channels:
            return {
                "status": "error",
                "message": f"Channel {channel_id} not found."
            }
        
        channel = self.secure_channels[channel_id]
        if channel["state"] != "established":
            return {
                "status": "error",
                "message": f"Channel {channel_id} is not established, cannot rotate keys."
            }
        
        try:
            # Generate a new key pair
            key_pair_result = await self.generate_key_pair()
            
            if key_pair_result["status"] != "success":
                return key_pair_result
            
            new_key_id = key_pair_result["key_id"]
            algorithm = key_pair_result["algorithm"]
            
            # Update channel with new key ID
            old_key_id = channel["key_id"]
            channel["key_id"] = new_key_id
            channel["algorithm"] = algorithm
            channel["state"] = "key_rotation_initiated"
            channel["last_activity"] = time.time()
            
            return {
                "status": "success",
                "channel_id": channel_id,
                "new_key_id": new_key_id,
                "old_key_id": old_key_id,
                "public_key_b64": key_pair_result["public_key_b64"],
                "algorithm": algorithm,
                "message": "Key rotation initiated. Exchange new keys with peer."
            }
        except Exception as e:
            logger.error(f"Error rotating channel keys: {e}")
            return {
                "status": "error",
                "message": f"Failed to rotate keys: {str(e)}"
            }
    
    # Additional utility methods
    
    async def import_public_key(self, public_key_b64: str, algorithm: str, peer_id: str) -> Dict[str, Any]:
        """Import a peer's public key"""
        try:
            public_key = base64.b64decode(public_key_b64)
            
            # Validate algorithm
            if algorithm not in self.supported_kem_algs and algorithm not in self.supported_sig_algs:
                if not LIBOQS_AVAILABLE:
                    # In fallback mode, we're more permissive
                    logger.warning(f"Unknown algorithm {algorithm}, but accepting in fallback mode.")
                else:
                    return {
                        "status": "error",
                        "message": f"Unsupported algorithm: {algorithm}"
                    }
            
            # Create key ID and store
            key_id = str(uuid.uuid4())
            self.key_pairs[key_id] = {
                "public_key": public_key,
                "algorithm": algorithm,
                "type": "imported",
                "peer_id": peer_id,
                "created_at": time.time()
            }
            
            return {
                "status": "success",
                "key_id": key_id,
                "algorithm": algorithm,
                "peer_id": peer_id
            }
        except Exception as e:
            logger.error(f"Error importing public key: {e}")
            return {
                "status": "error",
                "message": f"Failed to import public key: {str(e)}"
            }
    
    async def export_public_key(self, key_id: str) -> Dict[str, Any]:
        """Export a public key"""
        if key_id not in self.key_pairs:
            return {
                "status": "error",
                "message": f"Key ID {key_id} not found."
            }
        
        key_pair = self.key_pairs[key_id]
        if "public_key" not in key_pair:
            return {
                "status": "error",
                "message": f"No public key available for {key_id}."
            }
        
        return {
            "status": "success",
            "key_id": key_id,
            "algorithm": key_pair["algorithm"],
            "public_key_b64": base64.b64encode(key_pair["public_key"]).decode('ascii'),
            "created_at": key_pair["created_at"]
        }
    
    # Cleanup and maintenance
    
    async def cleanup_expired_resources(self, max_age_seconds: int = 86400) -> Dict[str, Any]:
        """Clean up expired keys, secrets, and channels"""
        current_time = time.time()
        
        # Clean up expired keys
        expired_keys = [k_id for k_id, k in self.key_pairs.items() 
                        if current_time - k["created_at"] > max_age_seconds]
        
        for key_id in expired_keys:
            if any(c.get("key_id") == key_id for c in self.secure_channels.values()):
                # Skip keys used by active channels
                continue
            
            del self.key_pairs[key_id]
        
        # Clean up expired shared secrets
        expired_secrets = [s_id for s_id, s in self.shared_secrets_store.items() 
                          if current_time - s["created_at"] > max_age_seconds]
        
        for secret_id in expired_secrets:
            # Skip secrets used by active channels
            if any("shared_secret" in c and c["shared_secret"] == self.shared_secrets_store[secret_id]["secret"] 
                  for c in self.secure_channels.values()):
                continue
            
            del self.shared_secrets_store[secret_id]
        
        # Clean up expired or inactive channels
        expired_channels = [c_id for c_id, c in self.secure_channels.items() 
                           if current_time - c["last_activity"] > max_age_seconds]
        
        for channel_id in expired_channels:
            del self.secure_channels[channel_id]
        
        return {
            "status": "success",
            "keys_cleaned": len(expired_keys),
            "secrets_cleaned": len(expired_secrets),
            "channels_cleaned": len(expired_channels)
        }

# Example usage (substantially revised for new channel flow)
async def example_usage():
    """Demonstrate usage of the QuantumResistantSecurity class with new channel flow"""
    
    qr_security = QuantumResistantSecurity()
    await qr_security.initialize(config_update={"enabled": True, "algorithms": {"key_exchange": "Kyber768"}}) # Ensure enabled
    print(f"Initial status: {await qr_security.get_security_status()}")

    # --- Secure Channel Establishment (Initiator A, Responder B) ---
    print("\n--- Secure Channel Establishment ---")
    device_A_id = "DeviceA_OpticalSwitch"
    device_B_id = "DeviceB_Controller"

    # 1. Device A (Initiator) starts channel establishment
    print(f"A ({device_A_id}): Initiating channel...")
    init_A_result = await qr_security.establish_secure_channel(device_id=device_A_id)
    print(f"A init result: {init_A_result}")
    if init_A_result["status"] != "success": return
    
    channel_id_A = init_A_result["channel_id"]
    pk_A_for_B_b64 = init_A_result["public_key_to_send_to_responder_b64"]
    kem_algo_A = init_A_result["algorithm"]

    # (Network: Device A sends pk_A_for_B_b64 and channel_id_A context (implicitly) to Device B)

    # 2. Device B (Responder) receives A's public key
    # For this example, Device B uses the same qr_security instance.
    # In reality, B would be a separate entity with its own QRS instance.
    print(f"\nB ({device_B_id}): Received PK from A. Responding...")
    # B specifies its device_id, and uses A's public key.
    resp_B_result = await qr_security.establish_secure_channel(device_id=device_B_id, initiator_public_key_b64=pk_A_for_B_b64)
    print(f"B response result: {resp_B_result}")
    if resp_B_result["status"] != "success": return

    channel_id_B = resp_B_result["channel_id"] # B gets its own channel_id for this interaction
    ct_B_for_A_b64 = resp_B_result["ciphertext_to_send_to_initiator_b64"]
    shared_secret_id_B = qr_security.secure_channel_store[channel_id_B].get("shared_secret_id") # B has the secret ID now
    print(f"B established its side of channel {channel_id_B} with shared_secret_id: {shared_secret_id_B}")

    # (Network: Device B sends ct_B_for_A_b64 back to Device A)

    # 3. Device A (Initiator) receives ciphertext from B and completes channel
    print(f"\nA ({device_A_id}): Received Ciphertext from B for channel {channel_id_A}. Completing...")
    complete_A_result = await qr_security.complete_secure_channel_initiator(channel_id_A, ct_B_for_A_b64)
    print(f"A completion result: {complete_A_result}")
    if complete_A_result["status"] != "success": return
    
    shared_secret_id_A = complete_A_result["shared_secret_id"]
    print(f"A established its side of channel {channel_id_A} with shared_secret_id: {shared_secret_id_A}")
    
    # Both A and B should now have the same shared secret (though potentially different IDs for it in their stores if not careful)
    # The shared_secret_id_A and shared_secret_id_B should point to the same secret material if KEM is correct.
    # For simplicity here, let's assume they now communicate using their respective channel IDs.

    # --- Secure Communication over Established Channels ---
    print("\n--- Secure Communication ---")
    message_from_A_bytes = b"Hello from Device A over PQC channel!"
    print(f"A sending on channel {channel_id_A}: '{message_from_A_bytes.decode()}'")
    send_A_comm_result = await qr_security.secure_channel_send(channel_id_A, message_from_A_bytes)
    if send_A_comm_result["status"] != "success": 
        print(f"A failed to send: {send_A_comm_result}")
        return

    print(f"A sent (ct_b64): {send_A_comm_result['ciphertext_base64'][:30]}..., nonce_b64: {send_A_comm_result['nonce_b64']}")
    
    # B receives this message on its channel_id_B
    print(f"B receiving on channel {channel_id_B}...")
    receive_B_comm_result = await qr_security.secure_channel_receive(
        channel_id_B, 
        send_A_comm_result["ciphertext_b64"], 
        send_A_comm_result["nonce_b64"]
    )
    if receive_B_comm_result["status"] == "success":
        received_on_B = receive_B_comm_result["plaintext_bytes"]
        print(f"B received: '{received_on_B.decode()}'")
        assert received_on_B == message_from_A_bytes
    else:
        print(f"B failed to receive: {receive_B_comm_result}")

    # --- Digital Signature Example ---
    print("\n--- Digital Signatures ---")
    data_to_sign_b64 = base64.b64encode(b"Critical network command.").decode('ascii')
    
    # Sign data (generates new sig key if ID not provided)
    sig_gen_res = await qr_security.generate_key_pair(key_type="sig_pair") # Generate a dedicated signature key
    if sig_gen_res["status"] != "success": print(f"Sig Key Gen failed: {sig_gen_res}"); return
    signing_key_id = sig_gen_res["key_id"]
    print(f"Generated signature key {signing_key_id} with algo {sig_gen_res['algorithm']}")

    sign_result = await qr_security.sign_data(data_to_sign_b64, signing_key_id=signing_key_id)
    print(f"Sign data result: {sign_result}")
    
    if sign_result["status"] == "success":
        signature_b64 = sign_result["signature_b64"]
        algo_used_sig = sign_result["algorithm_used"]
        
        # Verify signature (using the same key_id which implies using its public part)
        verify_result = await qr_security.verify_signature(data_to_sign_b64, signature_b64, signing_key_id)
        print(f"Verify signature result: {verify_result}")
        assert verify_result.get("verified") is True
        assert verify_result.get("algorithm_used") == algo_used_sig

    # --- Listing and Cleanup ---
    print("\n--- Listing and Cleanup ---")
    keys_list = await qr_security.list_keys()
    print(f"Keys stored: {len(keys_list.get('keys', []))}")
    secrets_list = await qr_security.list_shared_secrets()
    print(f"Shared Secrets stored: {len(secrets_list.get('shared_secrets', []))}")
    channels_list = await qr_security.list_secure_channels()
    print(f"Active Secure Channels: {len(channels_list.get('channels', []))}")
    
    print(f"Closing channel A ({channel_id_A})...")
    await qr_security.close_secure_channel(channel_id_A)
    print(f"Closing channel B ({channel_id_B})...")
    await qr_security.close_secure_channel(channel_id_B)
    
    # Final cleanup
    cleanup_result = await qr_security.cleanup_expired_resources(max_age_seconds=0) # Force cleanup for example
    print(f"Cleanup result: {cleanup_result}")
    print(f"Final status: {await qr_security.get_security_status()}")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(example_usage())
    finally:
        # Additional cleanup for asyncio if needed, e.g., cancelling tasks
        # For simple run_until_complete, this close is usually okay.
        loop.close()