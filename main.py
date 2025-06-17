# main.py - Optimized for multi-service environment
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn
import os
import json
import logging
from datetime import datetime
import httpx
import asyncio
from contextlib import asynccontextmanager
import psutil  # Add this for memory monitoring

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models
class SQLRequest(BaseModel):
    query_description: str = Field(..., description="Natural language description of the SQL query needed")
    database_schema: Optional[str] = Field(None, description="Database schema information (table names, columns)")
    database_type: Optional[str] = Field("postgresql", description="Database type (postgresql, mysql, sqlite)")
    table_names: Optional[List[str]] = Field(None, description="List of relevant table names")


class SQLResponse(BaseModel):
    sql_query: str
    explanation: str
    confidence: float
    generated_at: datetime
    warnings: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: str
    llm_available: bool
    model: str
    system_memory_usage: Optional[Dict[str, float]] = None


class LLMProvider:
    """Optimized LLM provider for resource-constrained environments"""

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Changed default model to lighter weight option
        self.model = os.getenv("OLLAMA_MODEL", "gemma:2b-instruct")
        # Reduced timeout for faster response in multi-service environment
        self.timeout = 30

    async def check_availability(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            async with httpx.AsyncClient(timeout=3) as client:  # Reduced timeout
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    return any(self.model in model["name"] for model in models)
                return False
        except Exception as e:
            logger.error(f"Error checking Ollama availability: {e}")
            return False

    def _build_prompt(self, description: str, schema: str = None, db_type: str = "postgresql",
                      table_names: List[str] = None) -> str:
        """Build optimized, concise prompt for SQL generation"""

        # More concise prompt to reduce token usage
        prompt = f"""Generate a {db_type.upper()} SQL query for: {description}

"""

        if schema:
            # Truncate schema if too long to save memory
            schema_truncated = schema[:500] + "..." if len(schema) > 500 else schema
            prompt += f"Schema: {schema_truncated}\n"

        if table_names:
            prompt += f"Tables: {', '.join(table_names[:5])}\n"  # Limit to 5 tables

        prompt += "Return only the SQL query with proper formatting:"

        return prompt

    async def generate_sql(self, description: str, schema: str = None, db_type: str = "postgresql",
                           table_names: List[str] = None) -> Dict[str, Any]:
        """Generate SQL query using LLM with memory optimizations"""

        # Check system memory before proceeding
        memory_info = psutil.virtual_memory()
        if memory_info.percent > 85:  # If memory usage > 85%
            logger.warning(f"High memory usage: {memory_info.percent}%")
            # Could implement queue or delay mechanism here

        try:
            prompt = self._build_prompt(description, schema, db_type, table_names)

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 500,  # Reduced from 1000
                    "num_ctx": 2048,  # Reduced context window
                    "num_predict": 200  # Limit prediction length
                }
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="LLM service unavailable"
                    )

                result = response.json()
                raw_response = result.get("response", "").strip()

                # Clean up the response
                sql_query = self._clean_sql_response(raw_response)

                # Generate simpler explanation to save resources
                explanation = await self._generate_simple_explanation(sql_query, description)

                # Calculate confidence
                confidence = self._calculate_confidence(sql_query, description)

                # Check for warnings
                warnings = self._check_warnings(sql_query)

                return {
                    "sql_query": sql_query,
                    "explanation": explanation,
                    "confidence": confidence,
                    "warnings": warnings
                }

        except httpx.TimeoutException:
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="LLM request timed out"
            )
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate SQL query"
            )

    def _clean_sql_response(self, response: str) -> str:
        """Clean and format the SQL response"""
        response = response.replace("```sql", "").replace("```", "")
        response = response.replace("SQL Query:", "").replace("Query:", "").strip()

        lines = [line.strip() for line in response.split('\n') if line.strip()]
        sql_query = '\n'.join(lines)

        if not sql_query.endswith(';'):
            sql_query += ';'

        return sql_query

    async def _generate_simple_explanation(self, sql_query: str, description: str) -> str:
        """Generate a simple explanation without additional LLM call to save resources"""
        # Instead of making another LLM call, provide rule-based explanation
        sql_upper = sql_query.upper()

        explanation_parts = []

        if 'SELECT' in sql_upper:
            if '*' in sql_query:
                explanation_parts.append("Selects all columns")
            else:
                explanation_parts.append("Selects specific columns")

        if 'JOIN' in sql_upper:
            join_count = sql_upper.count('JOIN')
            explanation_parts.append(f"Joins {join_count} table(s)")

        if 'WHERE' in sql_upper:
            explanation_parts.append("Filters results with conditions")

        if 'GROUP BY' in sql_upper:
            explanation_parts.append("Groups results")

        if 'ORDER BY' in sql_upper:
            explanation_parts.append("Sorts results")

        if explanation_parts:
            return f"This query {', '.join(explanation_parts).lower()} based on: {description}"
        else:
            return f"SQL query generated for: {description}"

    def _calculate_confidence(self, sql_query: str, description: str) -> float:
        """Calculate confidence score"""
        confidence = 0.5

        sql_keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY"]
        found_keywords = sum(1 for keyword in sql_keywords if keyword in sql_query.upper())
        confidence += min(found_keywords * 0.1, 0.3)

        if sql_query.count('(') == sql_query.count(')'):
            confidence += 0.1
        if sql_query.endswith(';'):
            confidence += 0.05
        if 'SELECT' in sql_query.upper() and 'FROM' in sql_query.upper():
            confidence += 0.1

        return min(confidence, 1.0)

    def _check_warnings(self, sql_query: str) -> List[str]:
        """Check for potential issues"""
        warnings = []
        sql_upper = sql_query.upper()

        if 'DELETE' in sql_upper and 'WHERE' not in sql_upper:
            warnings.append("DELETE without WHERE clause - will delete all rows!")

        if 'UPDATE' in sql_upper and 'WHERE' not in sql_upper:
            warnings.append("UPDATE without WHERE clause - will update all rows!")

        if 'DROP' in sql_upper:
            warnings.append("DROP statement detected!")

        return warnings


# Initialize LLM provider
llm_provider = LLMProvider()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting SQL Generator API...")

    # Log system memory info
    memory_info = psutil.virtual_memory()
    logger.info(
        f"System memory: {memory_info.total / (1024 ** 3):.1f}GB total, {memory_info.available / (1024 ** 3):.1f}GB available")

    is_available = await llm_provider.check_availability()
    if not is_available:
        logger.warning(f"LLM model {llm_provider.model} not available. Install with: ollama pull {llm_provider.model}")
    else:
        logger.info(f"LLM model {llm_provider.model} is available")

    yield

    logger.info("Shutting down SQL Generator API...")


# Initialize FastAPI app with reduced configuration for memory efficiency
app = FastAPI(
    title="SQL Generator API",
    description="Lightweight SQL query generator",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Specific origins for React
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only needed methods
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "SQL Generator API - Lightweight Mode"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with memory monitoring"""
    llm_available = await llm_provider.check_availability()

    # Add memory usage info
    memory_info = psutil.virtual_memory()
    system_memory = {
        "total_gb": round(memory_info.total / (1024 ** 3), 2),
        "available_gb": round(memory_info.available / (1024 ** 3), 2),
        "used_percent": memory_info.percent
    }

    return HealthResponse(
        status="healthy" if llm_available else "degraded",
        llm_available=llm_available,
        model=llm_provider.model,
        system_memory_usage=system_memory
    )


@app.post("/generate-sql", response_model=SQLResponse)
async def generate_sql(request: SQLRequest):
    """Generate SQL query - optimized for multi-service environment"""
    try:
        logger.info(f"Generating SQL for: {request.query_description[:100]}...")

        result = await llm_provider.generate_sql(
            description=request.query_description,
            schema=request.database_schema,
            db_type=request.database_type,
            table_names=request.table_names
        )

        response = SQLResponse(
            sql_query=result["sql_query"],
            explanation=result["explanation"],
            confidence=result["confidence"],
            generated_at=datetime.now(),
            warnings=result["warnings"]
        )

        logger.info(f"SQL generated (confidence: {result['confidence']:.2f})")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        workers=1
    )