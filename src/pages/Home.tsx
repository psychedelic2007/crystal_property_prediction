import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowRight, Zap, Shield, TrendingUp, Upload } from "lucide-react";
import Navigation from "@/components/Navigation";

const Home = () => {
  return (
    <div className="min-h-screen gradient-subtle">
      <Navigation />
      
      {/* Hero Section */}
      <section className="container mx-auto px-4 pt-20 pb-32">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="max-w-4xl mx-auto text-center"
        >
          <div className="inline-block mb-4 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium">
            AI-Powered Materials Science
          </div>
          
          <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            Predict Crystal Properties with Machine Learning
          </h1>
          
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Upload your crystal structure files or input atomic coordinates manually. 
            Get instant predictions with uncertainty quantification and detailed visualizations.
          </p>
          
          <div className="flex gap-4 justify-center flex-wrap">
            <Link to="/predict">
              <Button size="lg" className="gradient-primary text-lg px-8">
                Start Prediction
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </Link>
            <Link to="/about">
              <Button size="lg" variant="outline" className="text-lg px-8">
                Learn More
              </Button>
            </Link>
          </div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-4 pb-20">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.6 }}
          className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto"
        >
          <Card className="p-6 hover:shadow-lg transition-shadow">
            <div className="p-3 rounded-lg bg-primary/10 w-fit mb-4">
              <Upload className="h-6 w-6 text-primary" />
            </div>
            <h3 className="text-xl font-semibold mb-2">Easy Upload</h3>
            <p className="text-muted-foreground">
              Drag and drop CIF files or input structures manually with an intuitive interface.
            </p>
          </Card>

          <Card className="p-6 hover:shadow-lg transition-shadow">
            <div className="p-3 rounded-lg bg-primary/10 w-fit mb-4">
              <Zap className="h-6 w-6 text-primary" />
            </div>
            <h3 className="text-xl font-semibold mb-2">Fast Predictions</h3>
            <p className="text-muted-foreground">
              GPU-accelerated inference delivers results in seconds with state-of-the-art ML models.
            </p>
          </Card>

          <Card className="p-6 hover:shadow-lg transition-shadow">
            <div className="p-3 rounded-lg bg-primary/10 w-fit mb-4">
              <TrendingUp className="h-6 w-6 text-primary" />
            </div>
            <h3 className="text-xl font-semibold mb-2">Rich Insights</h3>
            <p className="text-muted-foreground">
              Interactive visualizations, uncertainty estimates, and downloadable reports.
            </p>
          </Card>
        </motion.div>
      </section>

      {/* How It Works */}
      <section className="container mx-auto px-4 pb-20">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.6 }}
          className="max-w-3xl mx-auto"
        >
          <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
          
          <div className="space-y-6">
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-10 h-10 rounded-full gradient-primary flex items-center justify-center text-white font-bold">
                1
              </div>
              <div>
                <h3 className="text-lg font-semibold mb-1">Upload Structure</h3>
                <p className="text-muted-foreground">
                  Provide your crystal structure via CIF file upload or manual coordinate input.
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0 w-10 h-10 rounded-full gradient-primary flex items-center justify-center text-white font-bold">
                2
              </div>
              <div>
                <h3 className="text-lg font-semibold mb-1">Model Processing</h3>
                <p className="text-muted-foreground">
                  Our ML model analyzes structural features and generates predictions with uncertainty quantification.
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0 w-10 h-10 rounded-full gradient-primary flex items-center justify-center text-white font-bold">
                3
              </div>
              <div>
                <h3 className="text-lg font-semibold mb-1">View Results</h3>
                <p className="text-muted-foreground">
                  Explore interactive visualizations, download reports, and track all your prediction jobs.
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="border-t bg-card/30 backdrop-blur-sm py-8">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>© 2025 CrystalPredict. Built for materials science research.</p>
        </div>
      </footer>
    </div>
  );
};

export default Home;
