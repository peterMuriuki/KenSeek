#!/usr/bin/env python3
"""
Database module for PDF extraction caching and tracking.
Uses SQLite to store OCR cache, extraction attempts, samples, and results.
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os


class ExtractionDatabase:
    """SQLite database for caching OCR and tracking extractions."""

    def __init__(self, db_path: str = "./extractions.db"):
        """
        Initialize database connection and create tables if needed.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        self._create_tables()

    def _create_tables(self):
        """Create all necessary tables if they don't exist."""
        cursor = self.conn.cursor()

        # PDFs table - track processed files
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pdfs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                file_hash TEXT UNIQUE NOT NULL,
                file_size INTEGER,
                num_pages INTEGER,
                ocr_date TIMESTAMP,
                ocr_model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # OCR cache - store extracted text per PDF
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ocr_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pdf_id INTEGER NOT NULL,
                page_number INTEGER,
                ocr_text TEXT NOT NULL,
                character_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (pdf_id) REFERENCES pdfs(id)
            )
        """)

        # Extraction attempts - one per run
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extraction_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pdf_id INTEGER NOT NULL,
                company_name TEXT,
                financial_year TEXT,
                model TEXT NOT NULL,
                num_samples INTEGER,
                extraction_method TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                success BOOLEAN,
                error_message TEXT,
                FOREIGN KEY (pdf_id) REFERENCES pdfs(id)
            )
        """)

        # Individual samples for self-consistency
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extraction_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id INTEGER NOT NULL,
                sample_number INTEGER,
                prompt_content TEXT,
                response_content TEXT,
                extracted_data TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (attempt_id) REFERENCES extraction_attempts(id)
            )
        """)

        # Final consensus results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extraction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                consensus_value TEXT,
                confidence TEXT,
                agreement_rate REAL,
                vote_distribution TEXT,
                FOREIGN KEY (attempt_id) REFERENCES extraction_attempts(id)
            )
        """)

        # Cost tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id INTEGER,
                service TEXT,
                model TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                estimated_cost REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (attempt_id) REFERENCES extraction_attempts(id)
            )
        """)

        # Create indexes for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pdf_hash
            ON pdfs(file_hash)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ocr_pdf
            ON ocr_cache(pdf_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_attempts_pdf
            ON extraction_attempts(pdf_id)
        """)

        self.conn.commit()

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """
        Compute SHA256 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            Hex string of file hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_pdf_by_hash(self, file_hash: str) -> Optional[Dict]:
        """
        Get PDF record by file hash.

        Args:
            file_hash: SHA256 hash of PDF file

        Returns:
            PDF record as dict, or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM pdfs WHERE file_hash = ?", (file_hash,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_or_create_pdf(self, file_path: str, file_hash: str,
                         num_pages: Optional[int] = None) -> int:
        """
        Get existing PDF record or create new one.

        Args:
            file_path: Path to PDF file
            file_hash: SHA256 hash of file
            num_pages: Number of pages in PDF

        Returns:
            PDF ID
        """
        # Check if exists
        pdf = self.get_pdf_by_hash(file_hash)
        if pdf:
            return pdf['id']

        # Create new record
        cursor = self.conn.cursor()
        file_size = os.path.getsize(file_path)

        cursor.execute("""
            INSERT INTO pdfs (file_path, file_hash, file_size, num_pages)
            VALUES (?, ?, ?, ?)
        """, (file_path, file_hash, file_size, num_pages))

        self.conn.commit()
        return cursor.lastrowid

    def cache_ocr_text(self, pdf_id: int, ocr_pages: List[str],
                       ocr_model: str = "gemini-2.5-flash-lite"):
        """
        Cache OCR text for a PDF.

        Args:
            pdf_id: PDF ID
            ocr_pages: List of OCR text per page
            ocr_model: Model used for OCR
        """
        cursor = self.conn.cursor()

        # Delete existing OCR cache for this PDF
        cursor.execute("DELETE FROM ocr_cache WHERE pdf_id = ?", (pdf_id,))

        # Insert new OCR pages
        for page_num, text in enumerate(ocr_pages, start=1):
            cursor.execute("""
                INSERT INTO ocr_cache (pdf_id, page_number, ocr_text, character_count)
                VALUES (?, ?, ?, ?)
            """, (pdf_id, page_num, text, len(text)))

        # Update PDF record
        cursor.execute("""
            UPDATE pdfs
            SET ocr_date = ?, ocr_model = ?, num_pages = ?
            WHERE id = ?
        """, (datetime.now(), ocr_model, len(ocr_pages), pdf_id))

        self.conn.commit()
        print(f"[DB] ✓ Cached OCR for {len(ocr_pages)} pages")

    def get_cached_ocr(self, pdf_id: int) -> Optional[List[str]]:
        """
        Get cached OCR text for a PDF.

        Args:
            pdf_id: PDF ID

        Returns:
            List of OCR text per page, or None if not cached
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT ocr_text
            FROM ocr_cache
            WHERE pdf_id = ?
            ORDER BY page_number
        """, (pdf_id,))

        rows = cursor.fetchall()
        if not rows:
            return None

        return [row['ocr_text'] for row in rows]

    def create_extraction_attempt(self, pdf_id: int, model: str,
                                 num_samples: int, extraction_method: str,
                                 company_name: Optional[str] = None,
                                 financial_year: Optional[str] = None) -> int:
        """
        Create new extraction attempt record.

        Args:
            pdf_id: PDF ID
            model: Model used for extraction
            num_samples: Number of samples for self-consistency
            extraction_method: "ocr_text" or "direct_image"
            company_name: Name of the company being analyzed
            financial_year: Financial year for the data being extracted

        Returns:
            Attempt ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO extraction_attempts
            (pdf_id, company_name, financial_year, model, num_samples, extraction_method, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (pdf_id, company_name, financial_year, model, num_samples, extraction_method, datetime.now()))

        self.conn.commit()
        return cursor.lastrowid

    def save_extraction_sample(self, attempt_id: int, sample_number: int,
                              prompt_content: str, response_content: str,
                              extracted_data: Dict, prompt_tokens: int,
                              completion_tokens: int):
        """
        Save an individual extraction sample.

        Args:
            attempt_id: Extraction attempt ID
            sample_number: Sample number (1, 2, 3, etc.)
            prompt_content: Full prompt sent to API
            response_content: Full response from API
            extracted_data: Parsed JSON data
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO extraction_samples
            (attempt_id, sample_number, prompt_content, response_content,
             extracted_data, prompt_tokens, completion_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (attempt_id, sample_number, prompt_content, response_content,
              json.dumps(extracted_data), prompt_tokens, completion_tokens))

        self.conn.commit()

    def save_extraction_results(self, attempt_id: int, consensus_data: Dict,
                               confidence_metrics: Dict):
        """
        Save final consensus extraction results.

        Args:
            attempt_id: Extraction attempt ID
            consensus_data: Final consensus values for each metric
            confidence_metrics: Confidence info for each metric
        """
        cursor = self.conn.cursor()

        for metric_name, consensus_value in consensus_data.items():
            confidence_info = confidence_metrics.get(metric_name, {})

            cursor.execute("""
                INSERT INTO extraction_results
                (attempt_id, metric_name, consensus_value, confidence,
                 agreement_rate, vote_distribution)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                attempt_id,
                metric_name,
                consensus_value,
                confidence_info.get('confidence_level', 'unknown'),
                confidence_info.get('confidence_score', 0.0),
                json.dumps(confidence_info.get('vote_distribution', {}))
            ))

        self.conn.commit()

    def complete_extraction_attempt(self, attempt_id: int, success: bool,
                                   error_message: Optional[str] = None):
        """
        Mark extraction attempt as completed.

        Args:
            attempt_id: Extraction attempt ID
            success: Whether extraction succeeded
            error_message: Error message if failed
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE extraction_attempts
            SET completed_at = ?, success = ?, error_message = ?
            WHERE id = ?
        """, (datetime.now(), success, error_message, attempt_id))

        self.conn.commit()

    def track_usage_cost(self, attempt_id: Optional[int], service: str,
                        model: str, prompt_tokens: int, completion_tokens: int,
                        estimated_cost: float):
        """
        Track API usage and estimated cost.

        Args:
            attempt_id: Extraction attempt ID (None for OCR)
            service: "gemini_ocr" or "deepseek_extraction"
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            estimated_cost: Estimated cost in USD
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO usage_costs
            (attempt_id, service, model, prompt_tokens, completion_tokens, estimated_cost)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (attempt_id, service, model, prompt_tokens, completion_tokens, estimated_cost))

        self.conn.commit()

    def get_extraction_history(self, pdf_id: int, limit: int = 10) -> List[Dict]:
        """
        Get extraction history for a PDF.

        Args:
            pdf_id: PDF ID
            limit: Maximum number of attempts to return

        Returns:
            List of extraction attempt records
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM extraction_attempts
            WHERE pdf_id = ?
            ORDER BY started_at DESC
            LIMIT ?
        """, (pdf_id, limit))

        return [dict(row) for row in cursor.fetchall()]

    def get_total_costs(self) -> Dict[str, float]:
        """
        Get total costs by service.

        Returns:
            Dict of total costs per service
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT service, SUM(estimated_cost) as total_cost
            FROM usage_costs
            GROUP BY service
        """)

        return {row['service']: row['total_cost'] for row in cursor.fetchall()}

    def get_all_successful_extractions(self, company_name: Optional[str] = None,
                                      financial_year: Optional[str] = None) -> List[Dict]:
        """
        Get all successful extraction attempts with their results.

        Args:
            company_name: Optional filter by company name
            financial_year: Optional filter by financial year

        Returns:
            List of extraction records with results and confidence metrics
        """
        cursor = self.conn.cursor()

        # Build query with optional filters
        query = """
            SELECT
                ea.id as attempt_id,
                ea.company_name,
                ea.financial_year,
                ea.model,
                ea.num_samples,
                ea.started_at,
                ea.completed_at,
                p.file_path,
                p.file_hash
            FROM extraction_attempts ea
            JOIN pdfs p ON ea.pdf_id = p.id
            WHERE ea.success = 1
        """
        params = []

        if company_name:
            query += " AND ea.company_name = ?"
            params.append(company_name)

        if financial_year:
            query += " AND ea.financial_year = ?"
            params.append(financial_year)

        query += " ORDER BY ea.company_name, ea.financial_year"

        cursor.execute(query, params)
        attempts = [dict(row) for row in cursor.fetchall()]

        # For each attempt, get the extraction results
        for attempt in attempts:
            attempt_id = attempt['attempt_id']

            # Get all extraction results for this attempt
            cursor.execute("""
                SELECT metric_name, consensus_value, confidence, agreement_rate, vote_distribution
                FROM extraction_results
                WHERE attempt_id = ?
            """, (attempt_id,))

            results = {}
            confidence_scores = {}

            for row in cursor.fetchall():
                metric_name = row['metric_name']
                results[metric_name] = row['consensus_value']
                confidence_scores[metric_name] = {
                    'confidence': row['confidence'],
                    'agreement_rate': row['agreement_rate']
                }

            attempt['extracted_data'] = results
            attempt['confidence_metrics'] = confidence_scores

        return attempts

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
