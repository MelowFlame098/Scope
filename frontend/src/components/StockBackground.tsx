import React, { useEffect, useRef } from 'react';

interface Point {
  x: number;
  y: number;
}

class ChartLine {
  color: string;
  width: number;
  points: Point[];
  speed: number;
  volatility: number;
  yOffset: number;
  canvasHeight: number;
  fill: boolean;

  constructor(color: string, width: number, speed: number, volatility: number, yOffset: number, canvasHeight: number, fill: boolean = false) {
    this.color = color;
    this.width = width;
    this.speed = speed;
    this.volatility = volatility;
    this.yOffset = yOffset;
    this.canvasHeight = canvasHeight;
    this.points = [];
    this.fill = fill;
  }

  init(canvasWidth: number) {
    this.points = [];
    let currentX = 0;
    let currentY = this.yOffset;
    
    // Fill initial points across the screen
    while (currentX < canvasWidth + 50) {
      this.points.push({ x: currentX, y: currentY });
      currentX += this.speed;
      currentY = this.nextY(currentY);
    }
  }

  nextY(prevY: number): number {
    let change = (Math.random() - 0.5) * this.volatility;
    let newY = prevY + change;
    
    // Keep within bounds (soft limits) with elastic pull to center
    const center = this.yOffset;
    const dist = newY - center;
    newY -= dist * 0.01; // Pull back to center

    return newY;
  }

  update(canvasWidth: number) {
    // Move all points left
    for (let i = 0; i < this.points.length; i++) {
      this.points[i].x -= 1; // Scroll speed
    }

    // Remove off-screen points
    if (this.points.length > 0 && this.points[0].x < -50) {
      this.points.shift();
    }

    // Add new points
    const lastPoint = this.points[this.points.length - 1];
    if (lastPoint.x < canvasWidth + 50) {
      this.points.push({
        x: lastPoint.x + this.speed,
        y: this.nextY(lastPoint.y)
      });
    }
  }

  draw(ctx: CanvasRenderingContext2D, canvasHeight: number) {
    if (this.points.length < 2) return;

    ctx.beginPath();
    ctx.moveTo(this.points[0].x, this.points[0].y);

    // Smooth curve using quadratic curves
    for (let i = 0; i < this.points.length - 1; i++) {
      const p0 = this.points[i];
      const p1 = this.points[i + 1];
      const midX = (p0.x + p1.x) / 2;
      const midY = (p0.y + p1.y) / 2;
      ctx.quadraticCurveTo(p0.x, p0.y, midX, midY);
    }
    
    // Connect to the last point
    const last = this.points[this.points.length - 1];
    ctx.lineTo(last.x, last.y);

    ctx.lineWidth = this.width;
    ctx.strokeStyle = this.color;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Add glow
    ctx.shadowBlur = 10;
    ctx.shadowColor = this.color;
    
    ctx.stroke();

    // Fill area if enabled
    if (this.fill) {
      ctx.lineTo(last.x, canvasHeight);
      ctx.lineTo(this.points[0].x, canvasHeight);
      ctx.closePath();
      
      const gradient = ctx.createLinearGradient(0, 0, 0, canvasHeight);
      gradient.addColorStop(0, this.color.replace(')', ', 0.2)').replace('rgb', 'rgba'));
      gradient.addColorStop(1, this.color.replace(')', ', 0.0)').replace('rgb', 'rgba'));
      
      ctx.fillStyle = gradient;
      ctx.fill();
    }
    
    // Reset shadow
    ctx.shadowBlur = 0;
  }
}

const StockBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let charts: ChartLine[] = [];

    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      
      // Re-initialize charts based on new height
      charts = [
        // Purple Area Chart (Main)
        new ChartLine('rgb(139, 92, 246)', 3, 20, 30, canvas.height * 0.6, canvas.height, true),
        // Green Line Chart (Secondary)
        new ChartLine('rgb(16, 185, 129)', 2, 30, 40, canvas.height * 0.4, canvas.height, false),
        // White Faint Line (Background noise)
        new ChartLine('rgba(255, 255, 255, 0.3)', 1, 40, 20, canvas.height * 0.5, canvas.height, false)
      ];
      
      charts.forEach(c => c.init(canvas.width));
    };

    window.addEventListener('resize', handleResize);
    handleResize();

    const drawGrid = () => {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        ctx.lineWidth = 1;
        
        // Vertical lines (moving)
        const time = Date.now() / 50;
        const spacing = 100;
        const offset = -(time % spacing);
        
        for (let x = offset; x < canvas.width; x += spacing) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }

        // Horizontal lines (static)
        for (let y = 0; y < canvas.height; y += spacing) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
        }
    };

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw dark background
      ctx.fillStyle = '#09090b'; // Zinc-950
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      drawGrid();

      charts.forEach(chart => {
        chart.update(canvas.width);
        chart.draw(ctx, canvas.height);
      });

      animationFrameId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: -1,
        pointerEvents: 'none'
      }}
    />
  );
};

export default StockBackground;
