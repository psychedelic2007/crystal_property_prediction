import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Atom } from "lucide-react";

const Navigation = () => {
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2 group">
            <div className="p-2 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
              <Atom className="h-6 w-6 text-primary" />
            </div>
            <span className="text-xl font-semibold">CrystalPredict</span>
          </Link>

          <div className="flex items-center gap-6">
            <Link to="/">
              <Button 
                variant={isActive("/") ? "default" : "ghost"}
                className="text-sm"
              >
                Home
              </Button>
            </Link>
            <Link to="/predict">
              <Button 
                variant={isActive("/predict") ? "default" : "ghost"}
                className="text-sm"
              >
                Submit Prediction
              </Button>
            </Link>
            <Link to="/about">
              <Button 
                variant={isActive("/about") ? "default" : "ghost"}
                className="text-sm"
              >
                About
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
