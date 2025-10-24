import { motion } from "framer-motion";
import Navigation from "@/components/Navigation";
import { Card } from "@/components/ui/card";
import { Atom, Database, Zap, Shield } from "lucide-react";

const About = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      <div className="container mx-auto px-4 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-4xl mx-auto"
        >
          <div className="mb-12 text-center">
            <h1 className="text-4xl font-bold mb-4">About CrystalPredict</h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              AI-powered platform for predicting crystal material properties with state-of-the-art machine learning models
            </p>
          </div>

          <div className="space-y-8 mb-12">
            <Card className="p-8">
              <h2 className="text-2xl font-semibold mb-4">Our Mission</h2>
              <p className="text-muted-foreground leading-relaxed">
                CrystalPredict accelerates materials discovery by providing researchers with instant, accurate predictions 
                of crystal properties. Our platform combines cutting-edge machine learning with robust computational infrastructure 
                to make advanced materials science accessible to everyone.
              </p>
            </Card>

            <div className="grid md:grid-cols-2 gap-6">
              <Card className="p-6">
                <div className="p-3 rounded-lg bg-primary/10 w-fit mb-4">
                  <Atom className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-lg font-semibold mb-2">Advanced ML Models</h3>
                <p className="text-muted-foreground text-sm">
                  Trained on extensive crystallographic databases with graph neural networks and ensemble methods 
                  for maximum accuracy and reliability.
                </p>
              </Card>

              <Card className="p-6">
                <div className="p-3 rounded-lg bg-primary/10 w-fit mb-4">
                  <Database className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-lg font-semibold mb-2">Comprehensive Data</h3>
                <p className="text-muted-foreground text-sm">
                  Our models are trained on millions of crystal structures from Materials Project, ICSD, 
                  and other authoritative databases.
                </p>
              </Card>

              <Card className="p-6">
                <div className="p-3 rounded-lg bg-primary/10 w-fit mb-4">
                  <Zap className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-lg font-semibold mb-2">Fast Inference</h3>
                <p className="text-muted-foreground text-sm">
                  GPU-accelerated predictions with TorchScript and ONNX optimization deliver results in seconds, 
                  not hours or days.
                </p>
              </Card>

              <Card className="p-6">
                <div className="p-3 rounded-lg bg-primary/10 w-fit mb-4">
                  <Shield className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-lg font-semibold mb-2">Uncertainty Quantification</h3>
                <p className="text-muted-foreground text-sm">
                  Every prediction includes uncertainty estimates using ensemble methods and Bayesian approaches 
                  for reliable confidence intervals.
                </p>
              </Card>
            </div>

            <Card className="p-8">
              <h2 className="text-2xl font-semibold mb-4">Technical Stack</h2>
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">Frontend</h4>
                  <p className="text-sm text-muted-foreground">
                    React + Vite + TypeScript + Tailwind CSS for a responsive, accessible interface with 
                    interactive 3D structure visualization using 3Dmol.js
                  </p>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Backend</h4>
                  <p className="text-sm text-muted-foreground">
                    FastAPI for high-performance REST APIs, Pymatgen for CIF parsing and structure analysis, 
                    Redis + Celery for asynchronous task processing
                  </p>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">ML Infrastructure</h4>
                  <p className="text-sm text-muted-foreground">
                    PyTorch models exported as TorchScript/ONNX for production deployment, with optional 
                    GPU acceleration and model versioning
                  </p>
                </div>
                <div>
                  <h4 className="font-semibold mb-2">Deployment</h4>
                  <p className="text-sm text-muted-foreground">
                    Docker containerization with Kubernetes orchestration for scalability, S3-compatible 
                    storage for files and results
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-8 bg-gradient-to-br from-primary/5 to-accent/5">
              <h2 className="text-2xl font-semibold mb-4">Model Information</h2>
              <p className="text-muted-foreground mb-4">
                Our current prediction model (v2.1.0) is trained to predict various crystal properties including:
              </p>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Band gap energy</li>
                <li>• Formation energy</li>
                <li>• Elastic constants and moduli</li>
                <li>• Stability metrics</li>
                <li>• Electronic structure properties</li>
              </ul>
              <p className="text-sm text-muted-foreground mt-4 italic">
                Model accuracy is continuously validated against experimental data and DFT calculations. 
                See our documentation for detailed benchmarks and limitations.
              </p>
            </Card>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default About;
