export const mockProjects = new Map()

const generateId = () => Math.random().toString(36).substring(2, 15)

const PIPELINE_STEPS = [
  { id: 'ocr', name: 'OCR' },
  { id: 'knowledge_extraction', name: 'KnowledgeExtraction' },
  { id: 'agent_discovery', name: 'AgentDiscovery' },
  { id: 'agent_deployment', name: 'AgentDeployment' },
  { id: 'chunking', name: 'Chunking' },
  { id: 'generation', name: 'DatasetGeneration' },
  { id: 'quality_control', name: 'QualityControl' },
  { id: 'fine_tuning', name: 'FineTuning' },
  { id: 'deployment', name: 'Deployment' },
]

export const mockState = {
  getProjects() {
    return Array.from(mockProjects.values()).sort((a, b) => 
      new Date(b.created_at) - new Date(a.created_at)
    )
  },

  getProject(id) {
    if (!mockProjects.has(id)) throw new Error('Project not found')
    return mockProjects.get(id)
  },

  createProject(data) {
    const id = generateId()
    const project = {
      id,
      name: data.name,
      description: data.description || '',
      mode: data.mode || 'full',
      model: data.model || 'openrouter/free',
      status: 'NOT_STARTED',
      created_at: new Date().toISOString(),
      logs: [],
      results: { step_results: {} },
      uploads: []
    }
    mockProjects.set(id, project)
    return project
  },

  startPipeline(id) {
    const project = this.getProject(id)
    if (project.status === 'RUNNING') return { status: 'already_running' }
    
    project.status = 'RUNNING'
    project.logs = [
      {
        id: generateId(),
        timestamp: new Date().toISOString(),
        type: 'ExecutionStarted',
        label: 'Pipeline execution initiated',
        summary: '',
        details: {}
      }
    ]
    
    // Start background simulation
    let stepIndex = 0
    const interval = setInterval(() => {
      // Allow early exit if project was deleted
      if (!mockProjects.has(id)) {
        clearInterval(interval)
        return
      }

      if (stepIndex >= PIPELINE_STEPS.length) {
        project.status = 'SUCCEEDED'
        project.logs.push({
          id: generateId(),
          timestamp: new Date().toISOString(),
          type: 'ExecutionSucceeded',
          label: 'Pipeline completed successfully',
          summary: '',
          details: {}
        })
        
        // Add final results
        project.results.virtual_mind_url = 'http://localhost:8000/chat/mock'
        project.results.dataset_download_url = 'http://localhost:8000/api/v1/mock/download'
        project.results.model_endpoint_url = 'https://api.modifai.io/v1/models/mock-123/invoke'
        project.results.virtual_mind_agents = [
            { id: 'hr-agent', name: 'HR Agent', endpoint: '/chat/hr-agent', role: 'Human Resources Policy Expert' },
            { id: 'legal-agent', name: 'Legal Agent', endpoint: '/chat/legal-agent', role: 'Contract & Compliance Analyst' }
        ]
        project.results.virtual_mind_automations = [
            { id: 'onboarding-flow', name: 'Employee Onboarding Flow', status: 'Deployed' },
            { id: 'contract-review', name: 'Contract Review Routing', status: 'Deployed' }
        ]
        project.results.n8n_url = 'http://localhost:5678/workflows/mock-123'
        
        clearInterval(interval)
        return
      }

      const step = PIPELINE_STEPS[stepIndex]
      
      // Enter step
      project.logs.push({
        id: generateId(),
        timestamp: new Date().toISOString(),
        type: 'TaskStateEntered',
        label: `Entered: ${step.name}`,
        summary: '',
        details: {}
      })
      
      // Wait 1 second then exit step
      setTimeout(() => {
        if (!mockProjects.has(id)) return
        project.results.step_results[step.id] = { mock_status: 'success' }
        project.logs.push({
          id: generateId(),
          timestamp: new Date().toISOString(),
          type: 'TaskStateExited',
          label: `Exited: ${step.name}`,
          summary: 'Completed successfully',
          details: {}
        })
      }, 1000)

      stepIndex++
    }, 2000)
    
    return { status: 'started' }
  },

  deleteProject(id) {
    mockProjects.delete(id)
    return { status: 'deleted' }
  }
}
