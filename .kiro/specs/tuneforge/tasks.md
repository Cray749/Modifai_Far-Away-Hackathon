# Implementation Plan: TuneForge

## Overview

This implementation plan breaks down the TuneForge platform into discrete, incremental coding tasks. The approach follows the pipeline architecture: document ingestion → chunking → intent selection → generation → curation → export → fine-tuning → comparison → deployment. Each task builds on previous work, with testing integrated throughout to validate functionality early.

The implementation uses Python with FastAPI for the backend, React for the frontend, and leverages the Hugging Face ecosystem (Transformers, PEFT) for LLM operations.

## Tasks

- [ ] 1. Set up project structure and core infrastructure
  - Create backend directory structure (app/, tests/, config/)
  - Create frontend directory structure (src/components/, src/pages/, src/api/)
  - Set up FastAPI application with CORS configuration
  - Configure environment variables for API keys and storage paths
  - Set up Redis for session management
  - Initialize testing frameworks (pytest for backend, Jest for frontend)
  - Create requirements.txt with core dependencies (fastapi, transformers, peft, hypothesis)
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [ ] 2. Implement document upload and text extraction
  - [ ] 2.1 Create DocumentUploadHandler class with file validation
    - Implement file format validation (PDF, TXT, DOCX only)
    - Implement file size validation (max 50MB)
    - Create UploadedFile and DocumentMetadata data models
    - _Requirements: 1.4, 1.5_
  
  - [ ] 2.2 Implement text extraction for each format
    - Implement PDF text extraction using pdfplumber
    - Implement TXT text extraction with encoding detection using chardet
    - Implement DOCX text extraction using python-docx
    - Store extracted text with document metadata
    - _Requirements: 1.1, 1.2, 1.3_
  
  - [ ] 2.3 Write property test for document extraction
    - **Property 1: Document extraction round-trip**
    - **Validates: Requirements 1.1, 1.2, 1.3**
  
  - [ ] 2.4 Write property test for format validation
    - **Property 2: Unsupported format rejection**
    - **Validates: Requirements 1.5**
  
  - [ ] 2.5 Create API endpoint for document upload
    - POST /api/documents/upload endpoint
    - Handle multipart form data
    - Return document metadata with unique ID
    - Implement error handling with user-friendly messages
    - _Requirements: 1.6, 1.7, 11.1, 11.2_
  
  - [ ] 2.6 Write property test for multiple document upload
    - **Property 3: Multiple document upload**
    - **Validates: Requirements 1.7**

- [ ] 3. Implement text chunking engine
  - [ ] 3.1 Create TextChunkingEngine class with tokenization
    - Implement tokenization using Hugging Face tokenizer
    - Create Chunk data model with metadata fields
    - Implement sentence boundary detection
    - _Requirements: 2.1, 2.6_
  
  - [ ] 3.2 Implement chunking algorithm with size constraints
    - Implement chunk splitting at sentence boundaries
    - Enforce target size range (200-1000 tokens)
    - Implement chunk merging for undersized chunks
    - Add overlap content between consecutive chunks (50 tokens)
    - _Requirements: 2.2, 2.3, 2.4, 2.5_
  
  - [ ] 3.3 Write property tests for chunking
    - **Property 4: Chunk size invariant**
    - **Property 5: Chunk boundary preservation**
    - **Property 6: Chunk overlap consistency**
    - **Property 7: Chunk metadata preservation**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**
  
  - [ ] 3.4 Create API endpoint for chunking
    - POST /api/documents/{id}/chunk endpoint
    - Store chunks in session storage
    - Return chunk count and preview
    - _Requirements: 2.6_

- [ ] 4. Implement intent selection system
  - [ ] 4.1 Create IntentSelector class and IntentType enum
    - Define IntentType enum with all supported intents
    - Create Intent data model with name, description, and template strategy
    - Implement intent storage in session
    - _Requirements: 3.1, 3.2, 3.3_
  
  - [ ] 4.2 Create API endpoints for intent management
    - GET /api/intents endpoint to list available intents
    - POST /api/session/{id}/intent endpoint to set selected intent
    - GET /api/session/{id}/intent endpoint to retrieve selected intent
    - _Requirements: 3.3, 3.4, 3.5_
  
  - [ ] 4.3 Write property tests for intent selection
    - **Property 8: Intent selection persistence**
    - **Property 9: Intent description completeness**
    - **Validates: Requirements 3.3, 3.4, 3.5**

- [ ] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement instruction-response generation
  - [ ] 6.1 Create InstructionGenerator class with LLM integration
    - Set up OpenAI or Anthropic API client
    - Create TrainingExample data model
    - Implement prompt template system for each intent type
    - _Requirements: 4.1, 4.7_
  
  - [ ] 6.2 Implement generation for each intent type
    - Create question-answering generation template and logic
    - Create summarization generation template and logic
    - Create tone-rewriting generation template and logic
    - Create classification generation template and logic
    - Create general-assistant generation template and logic
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.6_
  
  - [ ] 6.3 Implement generation orchestration with error handling
    - Implement batch generation across all chunks
    - Add retry logic with exponential backoff for API failures
    - Implement graceful degradation (continue on individual failures)
    - Validate generated examples (non-empty, proper structure)
    - Store successful examples with metadata
    - _Requirements: 4.8, 4.9, 11.4_
  
  - [ ] 6.4 Write property tests for generation
    - **Property 10: Training example generation**
    - **Property 11: Question-answering structure**
    - **Property 12: Summarization structure**
    - **Property 13: Generation error resilience**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.8, 4.9**
  
  - [ ] 6.5 Create API endpoint for generation
    - POST /api/session/{id}/generate endpoint
    - Implement as async task using Celery
    - Return job ID for status polling
    - Create GET /api/jobs/{id}/status endpoint for progress
    - _Requirements: 11.5_

- [ ] 7. Implement dataset review and editing
  - [ ] 7.1 Create DatasetEditor class with CRUD operations
    - Implement get_examples with pagination
    - Implement update_example with validation
    - Implement delete_example
    - Implement search_examples with text matching
    - Implement get_example_count
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_
  
  - [ ] 7.2 Write property tests for dataset editing
    - **Property 14: Example edit persistence**
    - **Property 15: Example deletion**
    - **Property 16: Example count accuracy**
    - **Property 17: Example search correctness**
    - **Validates: Requirements 5.3, 5.4, 5.5, 5.6, 5.7**
  
  - [ ] 7.3 Create API endpoints for dataset management
    - GET /api/session/{id}/examples endpoint with pagination
    - PUT /api/examples/{id} endpoint for updates
    - DELETE /api/examples/{id} endpoint
    - GET /api/session/{id}/examples/search endpoint
    - GET /api/session/{id}/examples/count endpoint
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

- [ ] 8. Implement JSONL export
  - [ ] 8.1 Create JSONLExporter class
    - Implement format_as_jsonl method
    - Implement filename generation with timestamp
    - Validate each line is valid JSON
    - Ensure instruction and response fields present
    - _Requirements: 6.1, 6.2, 6.3, 6.6_
  
  - [ ] 8.2 Write property tests for JSONL export
    - **Property 18: JSONL format validity**
    - **Property 19: JSONL export completeness**
    - **Property 20: Export filename format**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.6**
  
  - [ ] 8.3 Create API endpoint for export
    - GET /api/session/{id}/export endpoint
    - Return file download response
    - Set appropriate content-type and headers
    - _Requirements: 6.4, 6.5_

- [ ] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Implement base model selection
  - [ ] 10.1 Create model registry with metadata
    - Define supported base models (Mistral-7B, Llama-2-7B)
    - Create BaseModel data model with name, size, use cases
    - Implement model compatibility validation
    - Store selected model in session
    - _Requirements: 7.1, 7.2, 7.3, 7.5_
  
  - [ ] 10.2 Write property tests for model selection
    - **Property 21: Model selection persistence**
    - **Property 22: Model metadata completeness**
    - **Validates: Requirements 7.3, 7.4**
  
  - [ ] 10.3 Create API endpoints for model selection
    - GET /api/models endpoint to list available models
    - POST /api/session/{id}/model endpoint to set selected model
    - GET /api/session/{id}/model endpoint to retrieve selection
    - _Requirements: 7.1, 7.4_

- [ ] 11. Implement LoRA fine-tuning pipeline
  - [ ] 11.1 Create LoRAFineTuningPipeline class
    - Set up PEFT library integration
    - Define LoRA configuration (rank=8, alpha=16, target_modules)
    - Define training arguments (epochs=3, batch_size=4, lr=2e-4)
    - Implement 4-bit quantization for QLoRA
    - _Requirements: 8.2_
  
  - [ ] 11.2 Implement training execution
    - Implement model loading with quantization
    - Implement LoRA adapter application
    - Implement dataset loading and formatting
    - Implement trainer initialization and execution
    - Implement progress logging to Redis
    - Save adapter weights on completion
    - Store training metadata (duration, final loss)
    - _Requirements: 8.3, 8.4, 8.7_
  
  - [ ] 11.3 Implement training as async Celery task
    - Create Celery task for training
    - Implement job status tracking
    - Implement error handling with descriptive messages
    - Handle OOM errors with guidance
    - _Requirements: 8.5, 11.1, 11.2, 11.4_
  
  - [ ] 11.4 Write property tests for fine-tuning
    - **Property 23: Fine-tuning precondition validation**
    - **Property 24: Adapter persistence after training**
    - **Property 25: Training metadata capture**
    - **Validates: Requirements 8.1, 8.4, 8.7**
  
  - [ ] 11.5 Create API endpoints for fine-tuning
    - POST /api/session/{id}/finetune endpoint to start training
    - GET /api/jobs/{id}/status endpoint for training progress
    - GET /api/jobs/{id}/logs endpoint for training logs
    - _Requirements: 8.1, 8.3_

- [ ] 12. Implement model comparison interface
  - [ ] 12.1 Create ModelComparisonEngine class
    - Implement model loading (base and with adapter)
    - Implement response generation with proper parameters
    - Create ComparisonResult data model
    - Implement parallel generation for both models
    - _Requirements: 9.1, 9.2_
  
  - [ ] 12.2 Implement comparison with error handling
    - Handle generation failures per model independently
    - Label responses clearly (base vs tuned)
    - Support sequential prompts
    - _Requirements: 9.3, 9.6, 9.7_
  
  - [ ] 12.3 Write property tests for model comparison
    - **Property 26: Dual model response generation**
    - **Property 27: Response labeling**
    - **Property 28: Model error isolation**
    - **Property 29: Sequential prompt handling**
    - **Validates: Requirements 9.2, 9.3, 9.6, 9.7**
  
  - [ ] 12.4 Create API endpoints for comparison
    - POST /api/session/{id}/compare endpoint for generating comparison
    - Return both responses with labels and timing
    - _Requirements: 9.2, 9.3_

- [ ] 13. Implement model deployment and download
  - [ ] 13.1 Create ModelDeploymentHandler class
    - Implement adapter packaging (weights + config)
    - Generate loading instructions with code examples
    - Create download bundle as zip file
    - _Requirements: 10.1, 10.2, 10.4_
  
  - [ ] 13.2 Write property tests for deployment
    - **Property 30: Adapter package completeness**
    - **Property 31: Adapter loading instructions**
    - **Validates: Requirements 10.1, 10.2, 10.4**
  
  - [ ] 13.3 Create API endpoint for adapter download
    - GET /api/session/{id}/download endpoint
    - Return zip file with adapter and instructions
    - Handle download failures with retry option
    - _Requirements: 10.1, 10.3, 10.5_

- [ ] 14. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 15. Implement frontend UI components
  - [ ] 15.1 Create document upload page
    - File upload component with drag-and-drop
    - Format and size validation feedback
    - Upload progress indicator
    - Display uploaded documents list
    - _Requirements: 1.4, 1.5, 1.6_
  
  - [ ] 15.2 Create intent selection page
    - Display intent options with descriptions
    - Radio button or card selection UI
    - Store selection and navigate to next step
    - _Requirements: 3.1, 3.2, 3.5_
  
  - [ ] 15.3 Create generation progress page
    - Display generation progress
    - Show chunk processing status
    - Handle generation errors with user-friendly messages
    - Navigate to dataset review on completion
    - _Requirements: 4.8, 11.5_
  
  - [ ] 15.4 Create dataset review and editing page
    - Paginated table/list of training examples
    - Inline editing for instruction and response
    - Delete button for each example
    - Search/filter functionality
    - Example count display
    - Export button
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 6.5_
  
  - [ ] 15.5 Create model selection and fine-tuning page
    - Display available base models with metadata
    - Model selection UI
    - Start fine-tuning button with validation
    - Training progress display
    - Training logs viewer
    - _Requirements: 7.1, 7.2, 7.3, 8.1, 8.3_
  
  - [ ] 15.6 Create model comparison page
    - Split-screen layout for responses
    - Prompt input field
    - Generate button
    - Loading indicators per model
    - Clear labels for base vs tuned responses
    - Support for multiple sequential prompts
    - _Requirements: 9.1, 9.2, 9.3, 9.5, 9.7_
  
  - [ ] 15.7 Create deployment page
    - Download adapter button
    - Display loading instructions
    - Handle download errors
    - _Requirements: 10.1, 10.3, 10.5_

- [ ] 16. Implement session state management
  - [ ] 16.1 Create session management middleware
    - Generate unique session IDs
    - Store session data in Redis
    - Implement session expiration (24 hours)
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_
  
  - [ ] 16.2 Write property test for session persistence
    - **Property 34: Session state preservation**
    - **Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5**
  
  - [ ] 16.3 Implement navigation state preservation
    - Store pipeline stage in session
    - Allow navigation between stages
    - Preserve data when navigating
    - _Requirements: 12.5_

- [ ] 17. Implement comprehensive error handling
  - [ ] 17.1 Create error handling middleware
    - Catch all exceptions
    - Log errors with full details
    - Return user-friendly error messages
    - Avoid exposing stack traces
    - _Requirements: 11.1, 11.2, 11.4_
  
  - [ ] 17.2 Write property tests for error handling
    - **Property 32: Error message safety**
    - **Property 33: Error logging consistency**
    - **Validates: Requirements 11.1, 11.2, 11.4**
  
  - [ ] 17.3 Add error boundaries in frontend
    - Implement React error boundaries
    - Display user-friendly error pages
    - Provide retry options where applicable
    - _Requirements: 11.1, 11.3_

- [ ] 18. Integration and end-to-end testing
  - [ ] 18.1 Write end-to-end integration test
    - Test complete pipeline: upload → chunk → generate → edit → export
    - Test fine-tuning workflow: select model → train → compare
    - Verify session state preservation across stages
    - _Requirements: All requirements_
  
  - [ ] 18.2 Manual testing and bug fixes
    - Test all UI flows manually
    - Fix any discovered bugs
    - Verify error messages are user-friendly
    - Test on different file types and sizes

- [ ] 19. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation throughout development
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- The implementation uses Python with FastAPI for backend and React for frontend
- Frontend tasks can be parallelized with backend tasks after API contracts are defined
