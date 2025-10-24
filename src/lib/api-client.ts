// API client for FastAPI backend
// This connects the React frontend to your FastAPI model-serving backend

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'https://crystal-property-prediction.onrender.com';

export interface SubmitJobPayload {
  file?: File;
  structureData?: {
    latticeVectors: number[][];
    atomSites: Array<{
      species: string;
      x: number;
      y: number;
      z: number;
    }>;
  };
  metadata: {
    temperature?: string;
    pressure?: string;
    sampleId?: string;
  };
}

export interface JobStatus {
  jobId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: number;
  message?: string;
}

export interface PredictionResult {
  jobId: string;
  prediction: {
    formation_energy_per_atom: number;
    band_gap: number;
    stability: string;
    material_type: string;
    crystal_system: string;
    space_group_number: number;
  };
  structure_info: {
    formula: string;
    num_atoms: number;
    num_elements: number;
    density: number;
    volume: number;
    space_group_symbol: string;
    space_group_number: number;
    lattice_params: {
      a: number;
      b: number;
      c: number;
      alpha: number;
      beta: number;
      gamma: number;
    };
  };
  modelVersion: string;
  completedAt: string;
}

export interface CIFProcessingResult {
  success: boolean;
  structure_info: {
    formula: string;
    num_atoms: number;
    num_elements: number;
    density: number;
    volume: number;
    space_group_symbol: string;
    space_group_number: number;
    lattice_params: {
      a: number;
      b: number;
      c: number;
      alpha: number;
      beta: number;
      gamma: number;
    };
  };
  model_ready?: boolean;
  num_atoms?: number;
  num_edges?: number;
  space_group?: number;
  message?: string;
  error?: string;
}

class CrystalPredictAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Process CIF file and extract structure information
   */
  async processCIF(file: File): Promise<CIFProcessingResult> {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${this.baseUrl}/api/v1/process-cif`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorMessage = 'Failed to process CIF file';
        try {
          const error = await response.json();
          errorMessage = error.detail || errorMessage;
        } catch (e) {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      return response.json();
    } catch (error) {
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error('Network error: Unable to connect to backend server. Please ensure the backend is running on http://localhost:8000');
      }
      throw error;
    }
  }

  /**
   * Submit a new prediction job
   */
  async submitJob(payload: SubmitJobPayload): Promise<{ jobId: string }> {
    const formData = new FormData();

    if (payload.file) {
      formData.append('file', payload.file);
    }

    if (payload.structureData) {
      formData.append('structureData', JSON.stringify(payload.structureData));
    }

    formData.append('metadata', JSON.stringify(payload.metadata));

    const response = await fetch(`${this.baseUrl}/api/v1/submit`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to submit job');
    }

    return response.json();
  }

  /**
   * Get job status
   */
  async getJobStatus(jobId: string): Promise<JobStatus> {
    const response = await fetch(`${this.baseUrl}/api/v1/status/${jobId}`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to get job status');
    }

    return response.json();
  }

  /**
   * Get prediction results
   */
  async getResults(jobId: string): Promise<PredictionResult> {
    const response = await fetch(`${this.baseUrl}/api/v1/result/${jobId}`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to get results');
    }

    return response.json();
  }

  /**
   * Download result report as PDF
   */
  async downloadReport(jobId: string): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/v1/result/${jobId}/report`);

    if (!response.ok) {
      throw new Error('Failed to download report');
    }

    return response.blob();
  }

  /**
   * Get list of user's jobs
   */
  async getJobs(limit: number = 50): Promise<JobStatus[]> {
    const response = await fetch(`${this.baseUrl}/api/v1/jobs?limit=${limit}`);

    if (!response.ok) {
      throw new Error('Failed to get jobs');
    }

    return response.json();
  }
}

export const apiClient = new CrystalPredictAPI();
