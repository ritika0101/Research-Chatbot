"""
Enhanced Analyst - Information Extraction & Synthesis using SentenceTransformers and FAISS
"""

import os
import logging
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv
import groq
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class ExtractedInfo:
    """Structure for storing extracted information"""
    claim: str
    source_url: str
    confidence: float
    timestamp: str
    source_credibility: float  # 0-1 score based on domain authority
    context: str  # Surrounding text for context
    subq_id: str  # ID of the sub-question this answers

@dataclass
class Contradiction:
    """Structure for tracking contradicting information"""
    subq_id: str
    claims: List[ExtractedInfo]
    conflict_type: str  # e.g., "numerical", "factual", "temporal"
    severity: float  # 0-1 score of how severe the contradiction is

class AnalystV2:
    def __init__(self):
        self.setup_llm_and_embeddings()
        self.setup_vector_store()
        self.domain_authority = self._load_domain_authority()

    def setup_llm_and_embeddings(self):
        """Initialize LLM and embedding models"""
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("❌ GROQ_API_KEY not set in environment")

        # Initialize Groq LLM
        self.llm = Groq(
            api_key=groq_api_key,
            model_name="mixtral-8x7b-32768",
        )

        # Initialize Gemini embeddings
        self.embed_model = GeminiEmbedding(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name="embedding-001",
        )

    def setup_vector_store(self):
        """Initialize FAISS vector store and LlamaIndex components"""
        # Create FAISS vector store
        self.vector_store = FaissVectorStore(dim=768)  # Gemini embedding dimension
        
        # Create service context
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model,
        )

        # Initialize empty index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            service_context=self.service_context,
        )

    def _load_domain_authority(self) -> Dict[str, float]:
        """Load domain authority scores"""
        # You could load from a file, but here's a basic dict
        return {
            "uscis.gov": 1.0,
            "travel.state.gov": 0.95,
            "dhs.gov": 0.95,
            "bloomberg.com": 0.85,
            "reuters.com": 0.85,
            "wsj.com": 0.85,
            "nytimes.com": 0.85,
            "h1b.io": 0.75,
            "immi.org": 0.75,
            # Add more domains and their authority scores
        }

    def get_domain_authority(self, url: str) -> float:
        """Get domain authority score for a URL"""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        
        # Check exact domain first
        if domain in self.domain_authority:
            return self.domain_authority[domain]
            
        # Check for partial matches
        for known_domain, score in self.domain_authority.items():
            if known_domain in domain:
                return score * 0.9  # Slightly lower score for subdomains
                
        return 0.5  # Default score for unknown domains

    def process_documents(self, scout_results: List[Dict]) -> None:
        """Process and index scraped documents"""
        documents = []
        
        for result in scout_results:
            # Create metadata
            metadata = {
                "url": result["url"],
                "subq_id": result["subq_id"],
                "timestamp": result["fetched_at"],
                "domain_authority": self.get_domain_authority(result["url"]),
                "title": result["title"]
            }
            
            # Create Document object
            doc = Document(
                text=result["text"],
                metadata=metadata
            )
            documents.append(doc)
            
        # Parse into nodes and update index
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)
        self.index.refresh_ref_docs(nodes)
        
        logger.info(f"Processed and indexed {len(documents)} documents")

    def extract_info(self, subqs: List[Dict]) -> Tuple[List[ExtractedInfo], List[Contradiction]]:
        """
        Extract and synthesize information for each sub-question
        """
        extracted_info = []
        contradictions = []
        
        for subq in subqs:
            subq_id = subq["id"]
            query = self._generate_extraction_query(subq)
            
            # Query the vector store
            query_bundle = QueryBundle(query)
            retriever = self.index.as_retriever(similarity_top_k=5)
            nodes = retriever.retrieve(query_bundle)
            
            if not nodes:
                logger.warning(f"No relevant information found for {subq_id}")
                continue
                
            # Extract information using Groq
            results = self._extract_claims_from_nodes(nodes, subq)
            
            # Check for contradictions
            potential_contradictions = self._find_contradictions(results)
            if potential_contradictions:
                contradictions.extend(potential_contradictions)
                
            # Add verified results
            extracted_info.extend(results)
            
        return extracted_info, contradictions

    def _generate_extraction_query(self, subq: Dict) -> str:
        """Generate an optimized query for information extraction"""
        return f"""
        Find specific, factual information about: {subq['text']}
        Focus on:
        - Recent data (2024-2025)
        - Official numbers and statistics
        - Policy changes and updates
        - Verified facts from credible sources
        Ignore opinions and speculative content.
        """

    def _extract_claims_from_nodes(self, nodes: List[NodeWithScore], subq: Dict) -> List[ExtractedInfo]:
        """Extract specific claims from retrieved nodes using Groq"""
        results = []
        
        # Combine relevant context
        context = "\n".join([node.node.text for node in nodes])
        
        # Generate extraction prompt
        prompt = f"""
        Analyze this text and extract specific, factual claims about: {subq['text']}

        Text to analyze:
        {context}

        For each claim you find:
        1. State the claim precisely
        2. Rate your confidence (0-1)
        3. Note any uncertainties
        4. Include direct quotes where possible

        Format each claim as:
        CLAIM: [the claim]
        CONFIDENCE: [0-1 score]
        SOURCE_QUOTE: [relevant quote]
        ---
        """
        
        # Get structured extraction from Groq
        response = self.llm.complete(prompt)
        
        # Parse response and create ExtractedInfo objects
        claims = self._parse_extraction_response(response.text, nodes, subq["id"])
        results.extend(claims)
        
        return results

    def _parse_extraction_response(self, response: str, nodes: List[NodeWithScore], subq_id: str) -> List[ExtractedInfo]:
        """Parse LLM response into structured ExtractedInfo objects"""
        claims = []
        
        # Split into individual claims
        raw_claims = response.split("---")
        
        for raw_claim in raw_claims:
            if not raw_claim.strip():
                continue
                
            try:
                # Extract components using simple parsing
                claim_text = raw_claim.split("CLAIM:")[1].split("CONFIDENCE:")[0].strip()
                confidence = float(raw_claim.split("CONFIDENCE:")[1].split("\n")[0].strip())
                
                # Find best matching source node
                best_node = max(nodes, key=lambda n: n.score)
                
                # Create ExtractedInfo object
                info = ExtractedInfo(
                    claim=claim_text,
                    source_url=best_node.node.metadata["url"],
                    confidence=confidence,
                    timestamp=datetime.utcnow().isoformat(),
                    source_credibility=best_node.node.metadata["domain_authority"],
                    context=best_node.node.text[:200],  # First 200 chars for context
                    subq_id=subq_id
                )
                claims.append(info)
                
            except Exception as e:
                logger.warning(f"Error parsing claim: {e}")
                continue
                
        return claims

    def _find_contradictions(self, claims: List[ExtractedInfo]) -> List[Contradiction]:
        """Identify contradictions between claims"""
        contradictions = []
        
        # Group claims by sub-question
        by_subq = {}
        for claim in claims:
            if claim.subq_id not in by_subq:
                by_subq[claim.subq_id] = []
            by_subq[claim.subq_id].append(claim)
            
        # Check each group for contradictions
        for subq_id, subq_claims in by_subq.items():
            if len(subq_claims) < 2:
                continue
                
            # Compare claims pairwise
            for i in range(len(subq_claims)):
                for j in range(i + 1, len(subq_claims)):
                    conflict = self._check_contradiction(
                        subq_claims[i],
                        subq_claims[j]
                    )
                    
                    if conflict:
                        contradictions.append(Contradiction(
                            subq_id=subq_id,
                            claims=[subq_claims[i], subq_claims[j]],
                            conflict_type=conflict[0],
                            severity=conflict[1]
                        ))
                        
        return contradictions

    def _check_contradiction(self, claim1: ExtractedInfo, claim2: ExtractedInfo) -> Optional[Tuple[str, float]]:
        """
        Check if two claims contradict each other
        Returns (contradiction_type, severity) if found, None otherwise
        """
        # Create comparison prompt
        prompt = f"""
        Compare these two claims and identify if they contradict each other:

        Claim 1: {claim1.claim}
        Claim 2: {claim2.claim}

        If they contradict, explain:
        1. Type of contradiction (numerical, factual, temporal)
        2. Severity of contradiction (0-1)
        3. Brief explanation

        Format: TYPE|SEVERITY|EXPLANATION
        If no contradiction, respond with "NO_CONTRADICTION"
        """
        
        response = self.llm.complete(prompt).text.strip()
        
        if response == "NO_CONTRADICTION":
            return None
            
        try:
            conflict_type, severity, _ = response.split("|")
            return conflict_type, float(severity)
        except Exception:
            return None

def analyst_v2(scout_results: List[Dict], subqs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Main entry point for enhanced analysis
    """
    try:
        # Initialize analyzer
        analyzer = AnalystV2()
        
        # Process and index documents
        analyzer.process_documents(scout_results)
        
        # Extract information and find contradictions
        findings, contradictions = analyzer.extract_info(subqs)
        
        # Convert to dict format for compatibility
        findings_dict = [
            {
                "subq_id": f.subq_id,
                "claim": f.claim,
                "url": f.source_url,
                "confidence": f.confidence,
                "credibility": f.source_credibility
            }
            for f in findings
        ]
        
        contradictions_dict = [
            {
                "subq_id": c.subq_id,
                "claims": [cl.claim for cl in c.claims],
                "urls": [cl.source_url for cl in c.claims],
                "type": c.conflict_type,
                "severity": c.severity
            }
            for c in contradictions
        ]
        
        return findings_dict, contradictions_dict
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise RuntimeError(f"Analysis failed: {e}")

if __name__ == "__main__":
    # Test code
    test_results = [
        {
            "subq_id": "Q1",
            "url": "https://www.uscis.gov/example",
            "text": "H1B visa cap for 2025 remains at 65,000...",
            "title": "H1B Visa Updates 2025",
            "fetched_at": "2025-09-25T10:00:00Z"
        }
    ]
    
    test_subqs = [
        {
            "id": "Q1",
            "text": "H1B visa cap for 2025",
            "category": "statistics",
            "priority": "high"
        }
    ]
    
    findings, contradictions = analyst_v2(test_results, test_subqs)
    print("Findings:", findings)
    print("Contradictions:", contradictions)