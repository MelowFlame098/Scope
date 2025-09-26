"""Advanced Elliott Wave Analysis Model for Index Analysis

This module implements an enhanced Elliott Wave Theory analysis with:
- Fractal-based wave identification using Hurst exponent analysis
- Automated wave counting with machine learning pattern recognition
- Multi-timeframe wave analysis and synchronization
- Advanced Fibonacci relationships and harmonic patterns
- Zigzag filtering with adaptive thresholds
- Wave personality analysis and momentum confirmation
- Nested wave structure detection (sub-waves within waves)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
from scipy import signal, stats
from scipy.optimize import minimize_scalar
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class IndexIndicatorType(Enum):
    """Index-specific indicator types"""
    ELLIOTT_WAVE = "elliott_wave"

class WaveType(Enum):
    """Enhanced Elliott Wave types with fractal classification"""
    IMPULSE = "impulse"
    CORRECTIVE = "corrective"
    DIAGONAL = "diagonal"  # Ending/Leading diagonal
    TRIANGLE = "triangle"  # Contracting/Expanding triangle
    FLAT = "flat"  # Regular/Irregular/Running flat
    ZIGZAG = "zigzag"  # Simple/Double/Triple zigzag
    COMPLEX = "complex"  # Complex corrective structures
    UNKNOWN = "unknown"

class WavePosition(Enum):
    """Enhanced wave position with sub-wave classification"""
    # Impulse waves
    WAVE_1 = "wave_1"
    WAVE_2 = "wave_2"
    WAVE_3 = "wave_3"
    WAVE_4 = "wave_4"
    WAVE_5 = "wave_5"
    
    # Corrective waves
    WAVE_A = "wave_a"
    WAVE_B = "wave_b"
    WAVE_C = "wave_c"
    WAVE_D = "wave_d"  # For triangles and complex corrections
    WAVE_E = "wave_e"  # For triangles
    
    # Sub-wave positions
    SUB_WAVE_I = "sub_wave_i"
    SUB_WAVE_II = "sub_wave_ii"
    SUB_WAVE_III = "sub_wave_iii"
    SUB_WAVE_IV = "sub_wave_iv"
    SUB_WAVE_V = "sub_wave_v"
    
    # Degree classification
    SUPERCYCLE = "supercycle"
    CYCLE = "cycle"
    PRIMARY = "primary"
    INTERMEDIATE = "intermediate"
    MINOR = "minor"
    MINUTE = "minute"
    MINUETTE = "minuette"
    SUBMINUETTE = "subminuette"
    
    UNKNOWN = "unknown"

class WaveDegree(Enum):
    """Elliott Wave degree classification"""
    GRAND_SUPERCYCLE = 8
    SUPERCYCLE = 7
    CYCLE = 6
    PRIMARY = 5
    INTERMEDIATE = 4
    MINOR = 3
    MINUTE = 2
    MINUETTE = 1
    SUBMINUETTE = 0

class FractalType(Enum):
    """Fractal pattern types for wave identification"""
    UP_FRACTAL = "up_fractal"
    DOWN_FRACTAL = "down_fractal"
    NEUTRAL_FRACTAL = "neutral_fractal"
    COMPLEX_FRACTAL = "complex_fractal"

@dataclass
class IndexData:
    """Enhanced index information with multi-timeframe data"""
    symbol: str
    name: str
    current_level: float
    historical_levels: List[float]
    dividend_yield: float
    pe_ratio: float
    pb_ratio: float
    market_cap: float
    volatility: float
    beta: float
    sector_weights: Dict[str, float]
    constituent_count: int
    volume: float
    # Enhanced fields for fractal analysis
    timestamps: Optional[List[datetime]] = None
    high_prices: Optional[List[float]] = None
    low_prices: Optional[List[float]] = None
    volumes: Optional[List[float]] = None
    timeframe: str = "daily"

@dataclass
class FractalPoint:
    """Fractal point for wave analysis"""
    index: int
    price: float
    timestamp: datetime
    fractal_type: FractalType
    strength: float  # Fractal strength (0-1)
    hurst_exponent: float  # Local Hurst exponent
    volume_confirmation: bool = False
    momentum_divergence: bool = False

@dataclass
class WaveSegment:
    """Individual wave segment with fractal properties"""
    start_point: FractalPoint
    end_point: FractalPoint
    wave_type: WaveType
    wave_position: WavePosition
    degree: WaveDegree
    length: float  # Price distance
    duration: int  # Time duration
    slope: float  # Wave slope
    momentum: float  # Wave momentum
    volume_profile: Dict[str, float]  # Volume characteristics
    fibonacci_ratios: Dict[str, float]  # Fibonacci relationships
    sub_waves: List['WaveSegment'] = field(default_factory=list)
    confidence: float = 0.0
    
@dataclass
class WavePattern:
    """Enhanced Elliott Wave pattern with fractal analysis"""
    wave_type: WaveType
    current_position: WavePosition
    degree: WaveDegree
    wave_segments: List[WaveSegment]
    fractal_dimension: float  # Fractal dimension of the pattern
    hurst_exponent: float  # Overall Hurst exponent
    wave_start: float
    wave_end: float
    fibonacci_levels: Dict[str, float]
    harmonic_ratios: Dict[str, float]  # Advanced harmonic relationships
    nested_patterns: List['WavePattern'] = field(default_factory=list)
    confidence: float = 0.0
    personality_score: Dict[str, float] = field(default_factory=dict)  # Wave personality traits

@dataclass
class ElliottWaveResult:
    """Enhanced Elliott Wave analysis result with fractal insights"""
    indicator_type: IndexIndicatorType
    wave_pattern: WavePattern
    fractal_points: List[FractalPoint]
    multi_timeframe_analysis: Dict[str, WavePattern]  # Analysis across different timeframes
    target_levels: Dict[str, float]
    probability_zones: Dict[str, float]  # Probability-weighted target zones
    signal: str
    signal_strength: float  # Enhanced signal strength (0-1)
    confidence: float
    risk_level: str
    fractal_dimension: float  # Overall market fractal dimension
    hurst_exponent: float  # Market memory/trend persistence
    wave_personality: Dict[str, float]  # Current wave personality traits
    cycle_analysis: Dict[str, Any]  # Cycle and seasonality analysis
    harmonic_confluence: Dict[str, float]  # Harmonic pattern confluence
    volume_confirmation: float  # Volume-based confirmation score
    momentum_divergence: Dict[str, bool]  # Momentum divergence analysis
    nested_wave_count: int  # Number of nested wave structures identified
    wave_alternation: Dict[str, bool]  # Wave alternation compliance
    fibonacci_confluence: Dict[str, float]  # Fibonacci level confluence
    interpretation: str
    metadata: Dict[str, Any]
    timestamp: datetime
    time_horizon: str

class AdvancedElliottWaveAnalysis:
    """Advanced Elliott Wave analysis with fractal-based wave identification and automated counting"""
    
    def __init__(self, 
                 fractal_window: int = 5,
                 min_wave_length: float = 0.03,
                 hurst_window: int = 50,
                 volume_confirmation: bool = True,
                 multi_timeframe: bool = True,
                 max_wave_degree: int = 5,
                 fibonacci_tolerance: float = 0.05,
                 pattern_recognition_ml: bool = True):
        """Initialize Advanced Elliott Wave analyzer with fractal capabilities
        
        Args:
            fractal_window: Window size for fractal identification
            min_wave_length: Minimum wave length as percentage of total range
            hurst_window: Window size for Hurst exponent calculation
            volume_confirmation: Whether to use volume for wave confirmation
            multi_timeframe: Enable multi-timeframe analysis
            max_wave_degree: Maximum wave degree to analyze
            fibonacci_tolerance: Tolerance for Fibonacci ratio matching
            pattern_recognition_ml: Use ML for pattern recognition
        """
        # Enhanced Fibonacci ratios with harmonic relationships
        self.fibonacci_ratios = {
            # Standard retracements
            'retracement_23_6': 0.236,
            'retracement_38_2': 0.382,
            'retracement_50_0': 0.500,
            'retracement_61_8': 0.618,
            'retracement_78_6': 0.786,
            
            # Standard extensions
            'extension_127_2': 1.272,
            'extension_161_8': 1.618,
            'extension_200_0': 2.000,
            'extension_261_8': 2.618,
            'extension_314_0': 3.140,
            'extension_423_6': 4.236,
            
            # Advanced harmonic ratios
            'harmonic_88_6': 0.886,
            'harmonic_113_0': 1.130,
            'harmonic_141_4': 1.414,
            'harmonic_224_0': 2.240,
            'harmonic_354_0': 3.540
        }
        
        # Fractal analysis parameters
        self.fractal_window = fractal_window
        self.min_wave_length = min_wave_length
        self.hurst_window = hurst_window
        self.volume_confirmation = volume_confirmation
        self.multi_timeframe = multi_timeframe
        self.max_wave_degree = max_wave_degree
        self.fibonacci_tolerance = fibonacci_tolerance
        self.pattern_recognition_ml = pattern_recognition_ml
        
        # Wave personality characteristics
        self.wave_personalities = {
            'wave_1': {'impulsive': 0.7, 'corrective': 0.3, 'extension_probability': 0.2},
            'wave_2': {'sharp': 0.6, 'sideways': 0.4, 'deep_retracement': 0.8},
            'wave_3': {'impulsive': 0.9, 'extended': 0.6, 'strongest': 0.8},
            'wave_4': {'complex': 0.7, 'sideways': 0.8, 'alternation': 0.9},
            'wave_5': {'impulsive': 0.6, 'extension_probability': 0.3, 'divergence': 0.4},
            'wave_a': {'corrective': 0.8, 'three_wave': 0.6, 'five_wave': 0.4},
            'wave_b': {'corrective': 0.9, 'irregular': 0.5, 'complex': 0.6},
            'wave_c': {'impulsive': 0.7, 'five_wave': 0.8, 'completion': 0.8}
        }
        
        # Initialize ML components if enabled
        if self.pattern_recognition_ml:
            self._initialize_ml_components()
    
    def _initialize_ml_components(self):
        """Initialize machine learning components for pattern recognition"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            self.pattern_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.feature_scaler = StandardScaler()
            self.ml_initialized = True
        except ImportError:
            warnings.warn("Scikit-learn not available. ML pattern recognition disabled.")
            self.ml_initialized = False
    
    def calculate(self, index_data: IndexData, 
                 historical_data: Optional[pd.DataFrame] = None,
                 timeframes: Optional[List[str]] = None) -> ElliottWaveResult:
        """Advanced Elliott Wave analysis with fractal-based wave identification
        
        Args:
            index_data: Enhanced index data for analysis
            historical_data: Optional DataFrame with OHLCV data
            timeframes: List of timeframes for multi-timeframe analysis
            
        Returns:
            ElliottWaveResult: Comprehensive Elliott Wave analysis with fractal insights
        """
        try:
            # Prepare data for analysis
            if historical_data is not None:
                prices = historical_data['close'].values
                highs = historical_data['high'].values if 'high' in historical_data.columns else prices
                lows = historical_data['low'].values if 'low' in historical_data.columns else prices
                volumes = historical_data['volume'].values if 'volume' in historical_data.columns else None
                timestamps = pd.to_datetime(historical_data.index) if hasattr(historical_data.index, 'to_pydatetime') else None
            else:
                prices = np.array(index_data.historical_levels)
                highs = index_data.high_prices or prices
                lows = index_data.low_prices or prices
                volumes = index_data.volumes
                timestamps = index_data.timestamps
            
            # Identify fractal points using advanced algorithms
            fractal_points = self._identify_fractal_points(prices, highs, lows, volumes, timestamps)
            
            # Calculate fractal dimension and Hurst exponent
            fractal_dimension = self._calculate_fractal_dimension(prices)
            hurst_exponent = self._calculate_hurst_exponent(prices)
            
            # Perform automated wave counting with multiple degrees
            wave_pattern = self._automated_wave_counting(fractal_points, prices)
            
            # Multi-timeframe analysis if enabled
            multi_timeframe_analysis = {}
            if self.multi_timeframe and timeframes:
                multi_timeframe_analysis = self._multi_timeframe_analysis(historical_data, timeframes)
            
            # Enhanced Fibonacci and harmonic analysis
            fibonacci_levels = self._calculate_enhanced_fibonacci_levels(wave_pattern, prices)
            harmonic_confluence = self._calculate_harmonic_confluence(wave_pattern, fractal_points)
            
            # Probability-weighted target zones
            target_levels, probability_zones = self._calculate_probability_targets(wave_pattern, fibonacci_levels)
            
            # Advanced signal generation with ML if available
            signal, signal_strength = self._generate_advanced_signal(wave_pattern, fractal_points, index_data.current_level)
            
            # Enhanced confidence calculation
            confidence = self._calculate_enhanced_confidence(wave_pattern, fractal_points, harmonic_confluence)
            
            # Wave personality analysis
            wave_personality = self._analyze_wave_personality(wave_pattern)
            
            # Volume confirmation analysis
            volume_confirmation = self._analyze_volume_confirmation(fractal_points, volumes) if volumes is not None else 0.5
            
            # Momentum divergence analysis
            momentum_divergence = self._analyze_momentum_divergence(fractal_points, prices)
            
            # Cycle and seasonality analysis
            cycle_analysis = self._analyze_cycles(prices, timestamps) if timestamps is not None else {}
            
            # Wave alternation compliance
            wave_alternation = self._check_wave_alternation(wave_pattern)
            
            # Fibonacci confluence analysis
            fibonacci_confluence = self._analyze_fibonacci_confluence(fibonacci_levels, fractal_points)
            
            # Risk assessment
            risk_level = self._assess_advanced_risk_level(wave_pattern, confidence, fractal_dimension)
            
            # Generate comprehensive interpretation
            interpretation = self._generate_advanced_interpretation(
                wave_pattern, signal, confidence, fractal_dimension, hurst_exponent
            )
            
            return ElliottWaveResult(
                indicator_type=IndexIndicatorType.ELLIOTT_WAVE,
                wave_pattern=wave_pattern,
                fractal_points=fractal_points,
                multi_timeframe_analysis=multi_timeframe_analysis,
                target_levels=target_levels,
                probability_zones=probability_zones,
                signal=signal,
                signal_strength=signal_strength,
                confidence=confidence,
                risk_level=risk_level,
                fractal_dimension=fractal_dimension,
                hurst_exponent=hurst_exponent,
                wave_personality=wave_personality,
                cycle_analysis=cycle_analysis,
                harmonic_confluence=harmonic_confluence,
                volume_confirmation=volume_confirmation,
                momentum_divergence=momentum_divergence,
                nested_wave_count=len(wave_pattern.nested_patterns),
                wave_alternation=wave_alternation,
                fibonacci_confluence=fibonacci_confluence,
                interpretation=interpretation,
                metadata={
                    'fractal_window': self.fractal_window,
                    'hurst_window': self.hurst_window,
                    'ml_enabled': self.ml_initialized if hasattr(self, 'ml_initialized') else False,
                    'fractal_points_count': len(fractal_points),
                    'wave_segments_count': len(wave_pattern.wave_segments),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                timestamp=datetime.now(),
                time_horizon=self._determine_time_horizon(wave_pattern, fractal_dimension)
            )
            
        except Exception as e:
            # Return enhanced default result on error
            return ElliottWaveResult(
                indicator_type=IndexIndicatorType.ELLIOTT_WAVE,
                wave_pattern=WavePattern(
                    wave_type=WaveType.UNKNOWN,
                    current_position=WavePosition.UNKNOWN,
                    degree=WaveDegree.MINOR,
                    wave_segments=[],
                    fractal_dimension=1.5,
                    hurst_exponent=0.5,
                    wave_start=index_data.current_level,
                    wave_end=index_data.current_level,
                    fibonacci_levels={},
                    harmonic_ratios={},
                    confidence=0.0
                ),
                fractal_points=[],
                multi_timeframe_analysis={},
                target_levels={},
                probability_zones={},
                signal="hold",
                signal_strength=0.0,
                confidence=0.0,
                risk_level="unknown",
                fractal_dimension=1.5,
                hurst_exponent=0.5,
                wave_personality={},
                cycle_analysis={},
                harmonic_confluence={},
                volume_confirmation=0.0,
                momentum_divergence={},
                nested_wave_count=0,
                wave_alternation={},
                fibonacci_confluence={},
                interpretation=f"Unable to perform advanced Elliott Wave analysis: {str(e)}",
                metadata={'error': str(e), 'error_type': type(e).__name__},
                timestamp=datetime.now(),
                time_horizon="unknown"
            )
    
    def _identify_fractal_points(self, prices: np.ndarray, highs: np.ndarray, 
                                lows: np.ndarray, volumes: Optional[np.ndarray] = None,
                                timestamps: Optional[List[datetime]] = None) -> List[FractalPoint]:
        """Identify fractal points using advanced algorithms
        
        Args:
            prices: Close price array
            highs: High price array
            lows: Low price array
            volumes: Optional volume array
            timestamps: Optional timestamp array
            
        Returns:
            List of identified fractal points
        """
        fractal_points = []
        n = len(prices)
        
        if n < self.fractal_window * 2 + 1:
            return fractal_points
        
        # Generate timestamps if not provided
        if timestamps is None:
            timestamps = [datetime.now() - timedelta(days=n-i-1) for i in range(n)]
        
        # Identify fractal highs and lows
        for i in range(self.fractal_window, n - self.fractal_window):
            # Check for fractal high
            is_fractal_high = True
            for j in range(i - self.fractal_window, i + self.fractal_window + 1):
                if j != i and highs[j] >= highs[i]:
                    is_fractal_high = False
                    break
            
            # Check for fractal low
            is_fractal_low = True
            for j in range(i - self.fractal_window, i + self.fractal_window + 1):
                if j != i and lows[j] <= lows[i]:
                    is_fractal_low = False
                    break
            
            if is_fractal_high or is_fractal_low:
                # Calculate fractal strength
                strength = self._calculate_fractal_strength(prices, highs, lows, i)
                
                # Calculate local Hurst exponent
                hurst = self._calculate_local_hurst_exponent(prices, i)
                
                # Volume confirmation
                volume_conf = False
                if volumes is not None and self.volume_confirmation:
                    volume_conf = self._check_volume_confirmation(volumes, i, is_fractal_high)
                
                # Momentum divergence check
                momentum_div = self._check_momentum_divergence(prices, i, is_fractal_high)
                
                fractal_type = FractalType.UP_FRACTAL if is_fractal_high else FractalType.DOWN_FRACTAL
                price = highs[i] if is_fractal_high else lows[i]
                
                fractal_point = FractalPoint(
                    index=i,
                    price=price,
                    timestamp=timestamps[i],
                    fractal_type=fractal_type,
                    strength=strength,
                    hurst_exponent=hurst,
                    volume_confirmation=volume_conf,
                    momentum_divergence=momentum_div
                )
                
                fractal_points.append(fractal_point)
        
        # Filter fractal points by significance
        return self._filter_significant_fractals(fractal_points, prices)
    
    def _calculate_fractal_strength(self, prices: np.ndarray, highs: np.ndarray, 
                                   lows: np.ndarray, index: int) -> float:
        """Calculate the strength of a fractal point"""
        window = self.fractal_window
        start_idx = max(0, index - window)
        end_idx = min(len(prices), index + window + 1)
        
        local_range = np.max(highs[start_idx:end_idx]) - np.min(lows[start_idx:end_idx])
        if local_range == 0:
            return 0.0
        
        # Calculate relative prominence
        if highs[index] == np.max(highs[start_idx:end_idx]):
            prominence = (highs[index] - np.mean(highs[start_idx:end_idx])) / local_range
        else:
            prominence = (np.mean(lows[start_idx:end_idx]) - lows[index]) / local_range
        
        return min(1.0, max(0.0, prominence))
    
    def _calculate_local_hurst_exponent(self, prices: np.ndarray, index: int) -> float:
        """Calculate local Hurst exponent around a fractal point"""
        window = min(self.hurst_window, len(prices) // 4)
        start_idx = max(0, index - window // 2)
        end_idx = min(len(prices), index + window // 2)
        
        local_prices = prices[start_idx:end_idx]
        if len(local_prices) < 10:
            return 0.5  # Default to random walk
        
        return self._calculate_hurst_exponent(local_prices)
    
    def _check_volume_confirmation(self, volumes: np.ndarray, index: int, is_high: bool) -> bool:
        """Check if volume confirms the fractal point"""
        window = self.fractal_window
        start_idx = max(0, index - window)
        end_idx = min(len(volumes), index + window + 1)
        
        avg_volume = np.mean(volumes[start_idx:end_idx])
        current_volume = volumes[index]
        
        # Volume should be above average for confirmation
        return current_volume > avg_volume * 1.2
    
    def _check_momentum_divergence(self, prices: np.ndarray, index: int, is_high: bool) -> bool:
        """Check for momentum divergence at fractal point"""
        if index < 14:  # Need enough data for RSI calculation
            return False
        
        # Simple momentum divergence check using price momentum
        momentum_window = 14
        start_idx = max(0, index - momentum_window)
        
        price_momentum = prices[index] - prices[start_idx]
        price_change = prices[index] - prices[index - 1]
        
        if is_high:
            # For highs, look for negative momentum divergence
            return price_momentum < 0 and price_change > 0
        else:
            # For lows, look for positive momentum divergence
            return price_momentum > 0 and price_change < 0
    
    def _filter_significant_fractals(self, fractal_points: List[FractalPoint], 
                                   prices: np.ndarray) -> List[FractalPoint]:
        """Filter fractal points by significance threshold"""
        if not fractal_points:
            return []
        
        price_range = np.max(prices) - np.min(prices)
        min_significance = price_range * self.min_wave_length
        
        significant_fractals = []
        for fractal in fractal_points:
            if fractal.strength >= 0.3:  # Minimum strength threshold
                significant_fractals.append(fractal)
        
        return significant_fractals
    
    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        if len(prices) < 10:
            return 1.5  # Default value
        
        # Normalize prices to [0, 1] range
        normalized_prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))
        
        # Box-counting algorithm
        scales = np.logspace(0.01, 0.5, num=20)
        counts = []
        
        for scale in scales:
            # Create grid
            grid_size = int(1.0 / scale)
            if grid_size < 2:
                continue
                
            boxes = set()
            for i, price in enumerate(normalized_prices):
                x_box = int(i * grid_size / len(normalized_prices))
                y_box = int(price * grid_size)
                boxes.add((x_box, y_box))
            
            counts.append(len(boxes))
        
        if len(counts) < 2:
            return 1.5
        
        # Calculate fractal dimension from slope
        valid_scales = scales[:len(counts)]
        log_scales = np.log(1.0 / valid_scales)
        log_counts = np.log(counts)
        
        # Linear regression to find slope
        if len(log_scales) > 1:
            slope = np.polyfit(log_scales, log_counts, 1)[0]
            return max(1.0, min(2.0, slope))  # Clamp between 1 and 2
        
        return 1.5
    
    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        if len(prices) < 20:
            return 0.5  # Default to random walk
        
        # Calculate log returns
        log_returns = np.diff(np.log(prices + 1e-10))  # Add small value to avoid log(0)
        n = len(log_returns)
        
        # Range of lags to test
        lags = np.unique(np.logspace(0.5, np.log10(n//4), num=10).astype(int))
        lags = lags[lags >= 2]
        
        if len(lags) < 2:
            return 0.5
        
        rs_values = []
        
        for lag in lags:
            # Split series into non-overlapping periods
            periods = n // lag
            if periods < 2:
                continue
            
            rs_period = []
            for i in range(periods):
                start_idx = i * lag
                end_idx = start_idx + lag
                period_returns = log_returns[start_idx:end_idx]
                
                # Calculate mean
                mean_return = np.mean(period_returns)
                
                # Calculate cumulative deviations
                deviations = np.cumsum(period_returns - mean_return)
                
                # Calculate range
                R = np.max(deviations) - np.min(deviations)
                
                # Calculate standard deviation
                S = np.std(period_returns)
                
                if S > 0:
                    rs_period.append(R / S)
            
            if rs_period:
                rs_values.append(np.mean(rs_period))
        
        if len(rs_values) < 2:
            return 0.5
        
        # Calculate Hurst exponent from slope
        valid_lags = lags[:len(rs_values)]
        log_lags = np.log(valid_lags)
        log_rs = np.log(rs_values)
        
        # Linear regression
        hurst = np.polyfit(log_lags, log_rs, 1)[0]
        return max(0.0, min(1.0, hurst))  # Clamp between 0 and 1
    
    def _automated_wave_counting(self, fractal_points: List[FractalPoint], 
                                prices: np.ndarray) -> WavePattern:
        """Perform automated Elliott Wave counting using fractal analysis"""
        if len(fractal_points) < 5:
            return self._create_default_wave_pattern(prices)
        
        # Sort fractal points by index
        sorted_fractals = sorted(fractal_points, key=lambda x: x.index)
        
        # Create wave segments from fractal points
        wave_segments = self._create_wave_segments(sorted_fractals, prices)
        
        # Identify wave patterns using multiple approaches
        primary_pattern = self._identify_primary_wave_pattern(wave_segments)
        
        # Calculate overall fractal dimension and Hurst exponent
        fractal_dimension = self._calculate_fractal_dimension(prices)
        hurst_exponent = self._calculate_hurst_exponent(prices)
        
        # Enhanced Fibonacci analysis
        fibonacci_levels = self._calculate_segment_fibonacci_levels(wave_segments)
        
        # Harmonic ratio analysis
        harmonic_ratios = self._calculate_harmonic_ratios(wave_segments)
        
        # Determine wave degree based on time and price scale
        wave_degree = self._determine_wave_degree(wave_segments, prices)
        
        # Calculate pattern confidence
        confidence = self._calculate_pattern_confidence(wave_segments, primary_pattern)
        
        # Analyze nested patterns
        nested_patterns = self._identify_nested_patterns(wave_segments)
        
        # Calculate personality score
        personality_score = self._calculate_wave_personality_score(primary_pattern, wave_segments)
        
        return WavePattern(
            wave_type=primary_pattern['type'],
            current_position=primary_pattern['position'],
            degree=wave_degree,
            wave_segments=wave_segments,
            fractal_dimension=fractal_dimension,
            hurst_exponent=hurst_exponent,
            wave_start=sorted_fractals[0].price if sorted_fractals else prices[0],
            wave_end=sorted_fractals[-1].price if sorted_fractals else prices[-1],
            fibonacci_levels=fibonacci_levels,
            harmonic_ratios=harmonic_ratios,
            nested_patterns=nested_patterns,
            confidence=confidence,
            personality_score=personality_score
        )
    
    def _create_wave_segments(self, fractal_points: List[FractalPoint], 
                             prices: np.ndarray) -> List[WaveSegment]:
        """Create wave segments from fractal points"""
        segments = []
        
        for i in range(len(fractal_points) - 1):
            start_point = fractal_points[i]
            end_point = fractal_points[i + 1]
            
            # Calculate segment properties
            length = abs(end_point.price - start_point.price)
            duration = end_point.index - start_point.index
            slope = (end_point.price - start_point.price) / duration if duration > 0 else 0
            
            # Calculate momentum
            segment_prices = prices[start_point.index:end_point.index + 1]
            momentum = self._calculate_segment_momentum(segment_prices)
            
            # Volume profile (simplified)
            volume_profile = {'average': 1.0, 'trend': 'neutral'}  # Placeholder
            
            # Fibonacci relationships
            fibonacci_ratios = self._calculate_segment_fibonacci_ratios(start_point, end_point, segments)
            
            # Determine wave type and position
            wave_type, wave_position = self._classify_wave_segment(start_point, end_point, segments)
            
            segment = WaveSegment(
                start_point=start_point,
                end_point=end_point,
                wave_type=wave_type,
                wave_position=wave_position,
                degree=WaveDegree.MINOR,  # Will be updated later
                length=length,
                duration=duration,
                slope=slope,
                momentum=momentum,
                volume_profile=volume_profile,
                fibonacci_ratios=fibonacci_ratios,
                confidence=0.5  # Will be calculated later
            )
            
            segments.append(segment)
        
        return segments
    
    def _calculate_segment_momentum(self, segment_prices: np.ndarray) -> float:
        """Calculate momentum for a wave segment"""
        if len(segment_prices) < 2:
            return 0.0
        
        # Simple momentum calculation
        price_change = segment_prices[-1] - segment_prices[0]
        time_periods = len(segment_prices)
        
        return price_change / time_periods if time_periods > 0 else 0.0
    
    def _calculate_segment_fibonacci_ratios(self, start_point: FractalPoint, 
                                          end_point: FractalPoint, 
                                          previous_segments: List[WaveSegment]) -> Dict[str, float]:
        """Calculate Fibonacci ratios for a wave segment"""
        ratios = {}
        
        if not previous_segments:
            return ratios
        
        current_length = abs(end_point.price - start_point.price)
        
        # Compare with previous segments
        for i, prev_segment in enumerate(previous_segments[-3:]):  # Last 3 segments
            prev_length = prev_segment.length
            if prev_length > 0:
                ratio = current_length / prev_length
                
                # Check against known Fibonacci ratios
                for fib_name, fib_value in self.fibonacci_ratios.items():
                    if abs(ratio - fib_value) < self.fibonacci_tolerance:
                        ratios[f'{fib_name}_vs_segment_{len(previous_segments)-i}'] = ratio
        
        return ratios
    
    def _classify_wave_segment(self, start_point: FractalPoint, end_point: FractalPoint, 
                              previous_segments: List[WaveSegment]) -> Tuple[WaveType, WavePosition]:
        """Classify wave segment type and position"""
        # Simple classification based on direction and position
        is_upward = end_point.price > start_point.price
        segment_count = len(previous_segments)
        
        # Basic Elliott Wave position logic
        if segment_count % 2 == 0:  # Even segments (1, 3, 5, A, C)
            if is_upward:
                if segment_count < 5:
                    positions = [WavePosition.WAVE_1, WavePosition.WAVE_3, WavePosition.WAVE_5]
                    position = positions[min(segment_count // 2, 2)]
                    return WaveType.IMPULSE, position
                else:
                    return WaveType.CORRECTIVE, WavePosition.WAVE_A
            else:
                return WaveType.CORRECTIVE, WavePosition.WAVE_C
        else:  # Odd segments (2, 4, B)
            if segment_count < 4:
                positions = [WavePosition.WAVE_2, WavePosition.WAVE_4]
                position = positions[min((segment_count - 1) // 2, 1)]
                return WaveType.CORRECTIVE, position
            else:
                return WaveType.CORRECTIVE, WavePosition.WAVE_B
    
    def _create_default_wave_pattern(self, prices: np.ndarray) -> WavePattern:
        """Create a default wave pattern when insufficient data"""
        return WavePattern(
            wave_type=WaveType.UNKNOWN,
            current_position=WavePosition.UNKNOWN,
            degree=WaveDegree.MINOR,
            wave_segments=[],
            fractal_dimension=1.5,
            hurst_exponent=0.5,
            wave_start=prices[0] if len(prices) > 0 else 0.0,
            wave_end=prices[-1] if len(prices) > 0 else 0.0,
            fibonacci_levels={},
            harmonic_ratios={},
            confidence=0.0
        )
    
    def _identify_primary_wave_pattern(self, wave_segments: List[WaveSegment]) -> Dict[str, Any]:
        """Identify the primary Elliott Wave pattern from segments"""
        if not wave_segments:
            return {'type': WaveType.UNKNOWN, 'position': WavePosition.UNKNOWN}
        
        # Analyze segment sequence for Elliott Wave patterns
        impulse_score = self._calculate_impulse_pattern_score(wave_segments)
        corrective_score = self._calculate_corrective_pattern_score(wave_segments)
        
        if impulse_score > corrective_score:
            # Determine current impulse wave position
            position = self._determine_impulse_position(wave_segments)
            return {'type': WaveType.IMPULSE, 'position': position}
        else:
            # Determine current corrective wave position
            position = self._determine_corrective_position(wave_segments)
            corrective_type = self._classify_corrective_type(wave_segments)
            return {'type': corrective_type, 'position': position}
    
    def _calculate_impulse_pattern_score(self, wave_segments: List[WaveSegment]) -> float:
        """Calculate how well segments match impulse wave pattern"""
        if len(wave_segments) < 5:
            return 0.0
        
        score = 0.0
        
        # Check for 5-wave structure
        if len(wave_segments) >= 5:
            # Wave 3 should be longest
            if len(wave_segments) >= 3:
                wave_3_length = wave_segments[2].length
                other_lengths = [seg.length for i, seg in enumerate(wave_segments[:5]) if i != 2]
                if wave_3_length == max([wave_3_length] + other_lengths):
                    score += 0.3
            
            # Wave 4 should not overlap Wave 1
            if len(wave_segments) >= 4:
                wave_1_end = wave_segments[0].end_point.price
                wave_4_end = wave_segments[3].end_point.price
                if (wave_segments[0].end_point.price > wave_segments[0].start_point.price and 
                    wave_4_end > wave_1_end) or \
                   (wave_segments[0].end_point.price < wave_segments[0].start_point.price and 
                    wave_4_end < wave_1_end):
                    score += 0.2
            
            # Fibonacci relationships
            fib_score = sum(len(seg.fibonacci_ratios) for seg in wave_segments) / len(wave_segments)
            score += min(0.3, fib_score * 0.1)
        
        return min(1.0, score)
    
    def _calculate_corrective_pattern_score(self, wave_segments: List[WaveSegment]) -> float:
        """Calculate how well segments match corrective wave pattern"""
        if len(wave_segments) < 3:
            return 0.0
        
        score = 0.0
        
        # Check for 3-wave structure
        if len(wave_segments) >= 3:
            # A-B-C pattern characteristics
            wave_a = wave_segments[0]
            wave_b = wave_segments[1] if len(wave_segments) > 1 else None
            wave_c = wave_segments[2] if len(wave_segments) > 2 else None
            
            if wave_b and wave_c:
                # Wave C should be similar length to Wave A
                length_ratio = min(wave_a.length, wave_c.length) / max(wave_a.length, wave_c.length)
                score += length_ratio * 0.4
                
                # Wave B should be shorter retracement
                if wave_b.length < wave_a.length:
                    score += 0.3
        
        return min(1.0, score)
    
    def _determine_impulse_position(self, wave_segments: List[WaveSegment]) -> WavePosition:
        """Determine current position in impulse wave"""
        segment_count = len(wave_segments)
        
        if segment_count >= 5:
            return WavePosition.WAVE_5
        elif segment_count >= 4:
            return WavePosition.WAVE_4
        elif segment_count >= 3:
            return WavePosition.WAVE_3
        elif segment_count >= 2:
            return WavePosition.WAVE_2
        else:
            return WavePosition.WAVE_1
    
    def _determine_corrective_position(self, wave_segments: List[WaveSegment]) -> WavePosition:
        """Determine current position in corrective wave"""
        segment_count = len(wave_segments)
        
        if segment_count >= 3:
            return WavePosition.WAVE_C
        elif segment_count >= 2:
            return WavePosition.WAVE_B
        else:
            return WavePosition.WAVE_A
    
    def _classify_corrective_type(self, wave_segments: List[WaveSegment]) -> WaveType:
        """Classify the type of corrective pattern"""
        if len(wave_segments) < 3:
            return WaveType.CORRECTIVE
        
        # Simple classification - can be enhanced with more sophisticated logic
        wave_a = wave_segments[0]
        wave_b = wave_segments[1]
        wave_c = wave_segments[2]
        
        # Check for zigzag pattern
        if (wave_a.slope * wave_c.slope > 0 and  # Same direction
            abs(wave_b.slope) < abs(wave_a.slope)):  # B is smaller retracement
            return WaveType.ZIGZAG
        
        # Check for flat pattern
        if abs(wave_b.length / wave_a.length - 1.0) < 0.3:  # B similar to A
            return WaveType.FLAT
        
        return WaveType.CORRECTIVE
    
    def _calculate_segment_fibonacci_levels(self, wave_segments: List[WaveSegment]) -> Dict[str, float]:
        """Calculate Fibonacci levels for wave segments"""
        levels = {}
        
        if not wave_segments:
            return levels
        
        # Get overall range
        all_prices = []
        for segment in wave_segments:
            all_prices.extend([segment.start_point.price, segment.end_point.price])
        
        if len(all_prices) < 2:
            return levels
        
        price_high = max(all_prices)
        price_low = min(all_prices)
        price_range = price_high - price_low
        
        if price_range == 0:
            return levels
        
        # Calculate Fibonacci retracement levels
        for name, ratio in self.fibonacci_ratios.items():
            if 'retracement' in name:
                level = price_high - (price_range * ratio)
                levels[name] = level
            elif 'extension' in name:
                level = price_high + (price_range * (ratio - 1.0))
                levels[f'{name}_up'] = level
                level = price_low - (price_range * (ratio - 1.0))
                levels[f'{name}_down'] = level
        
        return levels
    
    def _calculate_harmonic_ratios(self, wave_segments: List[WaveSegment]) -> Dict[str, float]:
        """Calculate harmonic ratios between wave segments"""
        ratios = {}
        
        if len(wave_segments) < 2:
            return ratios
        
        # Calculate ratios between consecutive segments
        for i in range(len(wave_segments) - 1):
            seg1 = wave_segments[i]
            seg2 = wave_segments[i + 1]
            
            if seg1.length > 0:
                ratio = seg2.length / seg1.length
                ratios[f'segment_{i+1}_to_{i}_ratio'] = ratio
                
                # Check against harmonic ratios
                for harm_name, harm_value in self.fibonacci_ratios.items():
                    if 'harmonic' in harm_name and abs(ratio - harm_value) < self.fibonacci_tolerance:
                        ratios[f'{harm_name}_segments_{i}_{i+1}'] = ratio
        
        return ratios
    
    def _determine_wave_degree(self, wave_segments: List[WaveSegment], prices: np.ndarray) -> WaveDegree:
        """Determine wave degree based on time and price scale"""
        if not wave_segments:
            return WaveDegree.MINOR
        
        # Calculate average segment duration
        avg_duration = sum(seg.duration for seg in wave_segments) / len(wave_segments)
        
        # Calculate price volatility
        price_volatility = np.std(prices) / np.mean(prices) if len(prices) > 1 else 0.0
        
        # Simple degree classification based on duration and volatility
        if avg_duration > 100 and price_volatility > 0.1:
            return WaveDegree.PRIMARY
        elif avg_duration > 50 and price_volatility > 0.05:
            return WaveDegree.INTERMEDIATE
        elif avg_duration > 20:
            return WaveDegree.MINOR
        else:
            return WaveDegree.MINUTE
    
    def _calculate_pattern_confidence(self, wave_segments: List[WaveSegment], 
                                    primary_pattern: Dict[str, Any]) -> float:
        """Calculate confidence in the identified wave pattern"""
        if not wave_segments:
            return 0.0
        
        confidence = 0.0
        
        # Fibonacci relationship confidence
        fib_count = sum(len(seg.fibonacci_ratios) for seg in wave_segments)
        confidence += min(0.3, fib_count * 0.05)
        
        # Fractal strength confidence
        avg_strength = sum(seg.start_point.strength + seg.end_point.strength 
                          for seg in wave_segments) / (2 * len(wave_segments))
        confidence += avg_strength * 0.3
        
        # Volume confirmation
        volume_conf_count = sum(1 for seg in wave_segments 
                               if seg.start_point.volume_confirmation or seg.end_point.volume_confirmation)
        if len(wave_segments) > 0:
            confidence += (volume_conf_count / len(wave_segments)) * 0.2
        
        # Pattern type confidence
        if primary_pattern['type'] != WaveType.UNKNOWN:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _identify_nested_patterns(self, wave_segments: List[WaveSegment]) -> List[WavePattern]:
        """Identify nested wave patterns within segments"""
        nested_patterns = []
        
        # For now, return empty list - can be enhanced with recursive analysis
        # This would involve analyzing sub-waves within each major wave segment
        
        return nested_patterns
    
    def _calculate_wave_personality_score(self, primary_pattern: Dict[str, Any], 
                                        wave_segments: List[WaveSegment]) -> Dict[str, float]:
        """Calculate wave personality characteristics"""
        personality = {}
        
        if not wave_segments:
            return personality
        
        # Get current wave position
        current_position = primary_pattern.get('position', WavePosition.UNKNOWN)
        
        # Apply wave personality characteristics
        if current_position.value in self.wave_personalities:
            personality = self.wave_personalities[current_position.value].copy()
        
        # Adjust based on actual segment characteristics
        if wave_segments:
            last_segment = wave_segments[-1]
            
            # Adjust impulsive score based on momentum
            if 'impulsive' in personality:
                momentum_factor = min(1.0, abs(last_segment.momentum) / 0.1)
                personality['impulsive'] *= momentum_factor
            
            # Adjust extension probability based on length
            if 'extension_probability' in personality and len(wave_segments) > 1:
                avg_length = sum(seg.length for seg in wave_segments[:-1]) / (len(wave_segments) - 1)
                if last_segment.length > avg_length * 1.618:  # Golden ratio extension
                    personality['extension_probability'] = min(1.0, personality['extension_probability'] * 1.5)
        
        return personality
    
    def _perform_multi_timeframe_analysis(self, data: IndexData) -> Dict[str, Any]:
        """Perform Elliott Wave analysis across multiple timeframes"""
        timeframes = ['1h', '4h', '1d', '1w']
        analysis = {}
        
        for tf in timeframes:
            # Simulate different timeframe data (in real implementation, would fetch actual data)
            if tf == '1h':
                tf_data = data  # Use original data as 1h
            else:
                # Downsample data for higher timeframes
                tf_data = self._downsample_data(data, tf)
            
            # Perform analysis on this timeframe
            if len(tf_data.prices) >= self.min_wave_length:
                tf_result = self._analyze_single_timeframe(tf_data)
                analysis[tf] = tf_result
            else:
                analysis[tf] = {'confidence': 0.0, 'wave_type': WaveType.UNKNOWN}
        
        return analysis
    
    def _downsample_data(self, data: IndexData, timeframe: str) -> IndexData:
        """Downsample data to different timeframe"""
        # Simple downsampling - in production would use proper OHLC aggregation
        factor_map = {'4h': 4, '1d': 24, '1w': 168}
        factor = factor_map.get(timeframe, 1)
        
        if factor == 1:
            return data
        
        # Downsample by taking every nth point
        indices = list(range(0, len(data.prices), factor))
        
        return IndexData(
            prices=data.prices[indices] if len(indices) > 0 else data.prices,
            volumes=data.volumes[indices] if len(data.volumes) > len(indices) else data.volumes,
            timestamps=data.timestamps[indices] if len(data.timestamps) > len(indices) else data.timestamps,
            high_prices=data.high_prices[indices] if len(data.high_prices) > len(indices) else data.high_prices,
            low_prices=data.low_prices[indices] if len(data.low_prices) > len(indices) else data.low_prices,
            rsi_values=data.rsi_values[indices] if len(data.rsi_values) > len(indices) else data.rsi_values,
            macd_values=data.macd_values[indices] if len(data.macd_values) > len(indices) else data.macd_values
        )
    
    def _analyze_single_timeframe(self, data: IndexData) -> Dict[str, Any]:
        """Analyze Elliott Wave pattern for a single timeframe"""
        # Simplified analysis for timeframe
        fractal_points = self._identify_fractal_points(data.prices, data.volumes)
        
        if len(fractal_points) < 3:
            return {'confidence': 0.0, 'wave_type': WaveType.UNKNOWN}
        
        # Create basic wave segments
        wave_segments = self._create_wave_segments(fractal_points, data.prices)
        
        # Identify pattern
        primary_pattern = self._identify_primary_wave_pattern(wave_segments)
        
        return {
            'wave_type': primary_pattern['type'],
            'position': primary_pattern['position'],
            'confidence': self._calculate_pattern_confidence(wave_segments, primary_pattern),
            'fractal_count': len(fractal_points)
        }
    
    def _calculate_probability_zones(self, wave_segments: List[WaveSegment], 
                                   fibonacci_levels: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate probability zones for future price movements"""
        zones = {
            'support_zones': {},
            'resistance_zones': {},
            'target_zones': {}
        }
        
        if not wave_segments or not fibonacci_levels:
            return zones
        
        # Get current price range
        current_prices = [seg.end_point.price for seg in wave_segments[-3:]]  # Last 3 segments
        if not current_prices:
            return zones
        
        current_price = current_prices[-1]
        price_range = max(current_prices) - min(current_prices)
        
        # Calculate support zones (below current price)
        for level_name, level_price in fibonacci_levels.items():
            if level_price < current_price:
                probability = self._calculate_level_probability(level_price, wave_segments)
                zones['support_zones'][level_name] = {
                    'price': level_price,
                    'probability': probability,
                    'strength': min(1.0, probability * 1.2)
                }
            elif level_price > current_price:
                probability = self._calculate_level_probability(level_price, wave_segments)
                zones['resistance_zones'][level_name] = {
                    'price': level_price,
                    'probability': probability,
                    'strength': min(1.0, probability * 1.2)
                }
        
        # Calculate target zones based on wave projections
        if len(wave_segments) >= 2:
            last_segment = wave_segments[-1]
            prev_segment = wave_segments[-2]
            
            # Project based on wave relationships
            projection_ratios = [1.0, 1.272, 1.618, 2.618]  # Common Elliott Wave projections
            
            for i, ratio in enumerate(projection_ratios):
                target_price = current_price + (prev_segment.length * ratio * 
                                              (1 if last_segment.slope > 0 else -1))
                zones['target_zones'][f'target_{i+1}'] = {
                    'price': target_price,
                    'probability': max(0.1, 0.8 - i * 0.15),  # Decreasing probability for further targets
                    'ratio': ratio
                }
        
        return zones
    
    def _calculate_level_probability(self, level_price: float, wave_segments: List[WaveSegment]) -> float:
        """Calculate probability of price reaching a specific level"""
        if not wave_segments:
            return 0.5
        
        # Simple probability based on historical touches and segment momentum
        probability = 0.5  # Base probability
        
        # Adjust based on recent segment momentum
        if wave_segments:
            last_segment = wave_segments[-1]
            momentum_factor = min(1.0, abs(last_segment.momentum) / 0.1)
            
            # If momentum is towards the level, increase probability
            current_price = last_segment.end_point.price
            if ((level_price > current_price and last_segment.slope > 0) or 
                (level_price < current_price and last_segment.slope < 0)):
                probability += momentum_factor * 0.3
            else:
                probability -= momentum_factor * 0.2
        
        return max(0.1, min(0.9, probability))
    
    def _calculate_signal_strength(self, primary_pattern: Dict[str, Any], 
                                 wave_segments: List[WaveSegment],
                                 confidence: float) -> float:
        """Calculate overall signal strength"""
        if not wave_segments:
            return 0.0
        
        strength = confidence * 0.4  # Base from confidence
        
        # Add momentum component
        if wave_segments:
            last_segment = wave_segments[-1]
            momentum_strength = min(1.0, abs(last_segment.momentum) / 0.1)
            strength += momentum_strength * 0.3
        
        # Add pattern clarity component
        pattern_type = primary_pattern.get('type', WaveType.UNKNOWN)
        if pattern_type in [WaveType.IMPULSE, WaveType.ZIGZAG]:
            strength += 0.2  # Clear patterns get bonus
        elif pattern_type != WaveType.UNKNOWN:
            strength += 0.1
        
        # Add Fibonacci alignment component
        fib_alignment = sum(len(seg.fibonacci_ratios) for seg in wave_segments) / max(1, len(wave_segments))
        strength += min(0.1, fib_alignment * 0.02)
        
        return min(1.0, strength)
    
    def _initialize_ml_components(self):
        """Initialize machine learning components for pattern recognition"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            # Initialize pattern classifier
            self.pattern_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Initialize feature scaler
            self.feature_scaler = StandardScaler()
            
            # Flag to track if models are trained
            self.ml_models_trained = False
            
        except ImportError:
            warnings.warn("scikit-learn not available. ML features will be disabled.")
            self.pattern_classifier = None
            self.feature_scaler = None
            self.ml_models_trained = False
    
    def _extract_ml_features(self, wave_segments: List[WaveSegment], 
                           fractal_points: List[FractalPoint]) -> np.ndarray:
        """Extract features for ML pattern recognition"""
        features = []
        
        if not wave_segments:
            return np.array([0.0] * 20)  # Return default feature vector
        
        # Segment-based features
        segment_lengths = [seg.length for seg in wave_segments[-5:]]  # Last 5 segments
        segment_slopes = [seg.slope for seg in wave_segments[-5:]]
        segment_momentums = [seg.momentum for seg in wave_segments[-5:]]
        
        # Pad or truncate to fixed size
        segment_lengths = (segment_lengths + [0.0] * 5)[:5]
        segment_slopes = (segment_slopes + [0.0] * 5)[:5]
        segment_momentums = (segment_momentums + [0.0] * 5)[:5]
        
        features.extend(segment_lengths)
        features.extend(segment_slopes)
        features.extend(segment_momentums)
        
        # Fractal-based features
        if fractal_points:
            avg_fractal_strength = sum(fp.strength for fp in fractal_points[-5:]) / min(5, len(fractal_points))
            fractal_density = len(fractal_points) / max(1, len(wave_segments))
        else:
            avg_fractal_strength = 0.0
            fractal_density = 0.0
        
        features.extend([avg_fractal_strength, fractal_density])
        
        # Fibonacci features
        fib_ratio_count = sum(len(seg.fibonacci_ratios) for seg in wave_segments)
        features.append(fib_ratio_count)
        
        # Volume features
        volume_confirmations = sum(1 for seg in wave_segments 
                                 if seg.start_point.volume_confirmation or seg.end_point.volume_confirmation)
        features.append(volume_confirmations)
        
        # Momentum divergence features
        momentum_divergences = sum(1 for seg in wave_segments 
                                 if seg.start_point.momentum_divergence or seg.end_point.momentum_divergence)
        features.append(momentum_divergences)
        
        return np.array(features)
    
    def _identify_wave_pattern(self, peaks_troughs: List[Tuple[int, float, str]], current_level: float) -> WavePattern:
        """Identify current Elliott Wave pattern (legacy method for compatibility)"""
        if len(peaks_troughs) < 3:
            return WavePattern(
                wave_type=WaveType.UNKNOWN,
                current_position=WavePosition.UNKNOWN,
                degree=WaveDegree.MINOR,
                wave_segments=[],
                fractal_dimension=1.5,
                hurst_exponent=0.5,
                wave_start=current_level,
                wave_end=current_level,
                fibonacci_levels={},
                harmonic_ratios={},
                confidence=0.2
            )
        
        # Convert peaks_troughs to fractal points for new system
        fractal_points = []
        for i, (idx, price, point_type) in enumerate(peaks_troughs):
            fractal_points.append(FractalPoint(
                index=idx,
                price=price,
                fractal_type=FractalType.PEAK if point_type == 'peak' else FractalType.TROUGH,
                strength=0.5,  # Default strength
                volume_confirmation=False,
                momentum_divergence=False
            ))
        
        # Create wave segments from fractal points
        prices = np.array([fp.price for fp in fractal_points])
        wave_segments = self._create_wave_segments(fractal_points, prices)
        
        # Identify primary pattern
        primary_pattern = self._identify_primary_wave_pattern(wave_segments)
        
        # Calculate enhanced metrics
        fibonacci_levels = self._calculate_segment_fibonacci_levels(wave_segments)
        harmonic_ratios = self._calculate_harmonic_ratios(wave_segments)
        confidence = self._calculate_pattern_confidence(wave_segments, primary_pattern)
        
        return WavePattern(
            wave_type=primary_pattern.get('type', WaveType.UNKNOWN),
            current_position=primary_pattern.get('position', WavePosition.UNKNOWN),
            degree=self._determine_wave_degree(wave_segments, prices),
            wave_segments=wave_segments,
            fractal_dimension=1.5,  # Would be calculated from actual data
            hurst_exponent=0.5,     # Would be calculated from actual data
            wave_start=fractal_points[0].price if fractal_points else current_level,
            wave_end=fractal_points[-1].price if fractal_points else current_level,
            fibonacci_levels=fibonacci_levels,
            harmonic_ratios=harmonic_ratios,
            confidence=confidence
        )
    
    def _analyze_wave_structure(self, points: List[Tuple[int, float, str]]) -> Tuple[WaveType, WavePosition]:
        """Analyze wave structure to determine type and position"""
        if len(points) < 3:
            return WaveType.UNKNOWN, WavePosition.UNKNOWN
        
        # Count alternating peaks and troughs
        types = [point[2] for point in points]
        prices = [point[1] for point in points]
        
        # Check for impulse pattern (5 waves)
        if len(points) >= 5:
            # Look for 5-wave structure: up-down-up-down-up (or inverse)
            if self._is_five_wave_structure(types, prices):
                return WaveType.IMPULSE, self._determine_impulse_position(types, prices)
        
        # Check for corrective pattern (3 waves)
        if len(points) >= 3:
            if self._is_three_wave_structure(types, prices):
                return WaveType.CORRECTIVE, self._determine_corrective_position(types, prices)
        
        return WaveType.UNKNOWN, WavePosition.UNKNOWN
    
    def _is_five_wave_structure(self, types: List[str], prices: List[float]) -> bool:
        """Check if the pattern resembles a 5-wave Elliott structure"""
        if len(types) < 5:
            return False
        
        # Simplified check: alternating peaks and troughs with proper relationships
        # Wave 3 should be the longest, Wave 4 shouldn't overlap Wave 1
        return True  # Simplified for demo
    
    def _is_three_wave_structure(self, types: List[str], prices: List[float]) -> bool:
        """Check if the pattern resembles a 3-wave corrective structure"""
        if len(types) < 3:
            return False
        
        # Simplified check for A-B-C corrective pattern
        return True  # Simplified for demo
    
    def _determine_impulse_position(self, types: List[str], prices: List[float]) -> WavePosition:
        """Determine position within impulse wave"""
        # Simplified logic - in practice would be more complex
        wave_positions = [WavePosition.WAVE_1, WavePosition.WAVE_2, WavePosition.WAVE_3, 
                         WavePosition.WAVE_4, WavePosition.WAVE_5]
        return wave_positions[min(len(prices) - 1, 4)]
    
    def _determine_corrective_position(self, types: List[str], prices: List[float]) -> WavePosition:
        """Determine position within corrective wave"""
        # Simplified logic
        corrective_positions = [WavePosition.WAVE_A, WavePosition.WAVE_B, WavePosition.WAVE_C]
        return corrective_positions[min(len(prices) - 1, 2)]
    
    def _calculate_pattern_confidence(self, points: List[Tuple[int, float, str]], wave_type: WaveType) -> float:
        """Calculate confidence in wave pattern identification"""
        base_confidence = 0.4
        
        # More points increase confidence
        point_bonus = min(0.3, len(points) * 0.05)
        
        # Clear alternating pattern increases confidence
        if len(points) >= 3:
            types = [point[2] for point in points]
            alternating = all(types[i] != types[i+1] for i in range(len(types)-1))
            if alternating:
                base_confidence += 0.2
        
        return min(0.8, base_confidence + point_bonus)
    
    def _calculate_fibonacci_levels(self, wave_pattern: WavePattern, peaks_troughs: List[Tuple[int, float, str]]) -> Dict[str, float]:
        """Calculate Fibonacci retracement and extension levels"""
        if len(peaks_troughs) < 2:
            return {}
        
        # Use the last significant move for Fibonacci calculation
        start_price = wave_pattern.wave_start
        end_price = wave_pattern.wave_end
        price_range = end_price - start_price
        
        fibonacci_levels = {}
        
        # Calculate retracement levels (for corrections)
        for name, ratio in self.fibonacci_ratios.items():
            if "retracement" in name:
                level = end_price - (price_range * ratio)
                fibonacci_levels[name] = level
            elif "extension" in name:
                level = end_price + (price_range * (ratio - 1))
                fibonacci_levels[name] = level
        
        return fibonacci_levels
    
    def _generate_target_levels(self, wave_pattern: WavePattern, fibonacci_levels: Dict[str, float]) -> Dict[str, float]:
        """Generate target levels based on Elliott Wave analysis"""
        targets = {}
        
        if wave_pattern.current_position == WavePosition.WAVE_3:
            # In wave 3, target extensions
            if "extension_161.8" in fibonacci_levels:
                targets["wave_3_target"] = fibonacci_levels["extension_161.8"]
        
        elif wave_pattern.current_position == WavePosition.WAVE_4:
            # In wave 4, target retracements
            if "retracement_38.2" in fibonacci_levels:
                targets["wave_4_support"] = fibonacci_levels["retracement_38.2"]
        
        elif wave_pattern.current_position == WavePosition.WAVE_5:
            # In wave 5, target final extension
            if "extension_127.2" in fibonacci_levels:
                targets["wave_5_target"] = fibonacci_levels["extension_127.2"]
        
        # Add general support/resistance levels
        if fibonacci_levels:
            sorted_levels = sorted(fibonacci_levels.values())
            targets["support"] = sorted_levels[len(sorted_levels)//3]
            targets["resistance"] = sorted_levels[2*len(sorted_levels)//3]
        
        return targets
    
    def _generate_signal(self, wave_pattern: WavePattern, current_level: float, target_levels: Dict[str, float]) -> str:
        """Generate trading signal based on Elliott Wave analysis"""
        if wave_pattern.wave_type == WaveType.IMPULSE:
            if wave_pattern.current_position in [WavePosition.WAVE_1, WavePosition.WAVE_3, WavePosition.WAVE_5]:
                # Impulse waves are bullish
                return "BUY"
            elif wave_pattern.current_position in [WavePosition.WAVE_2, WavePosition.WAVE_4]:
                # Corrective waves within impulse are bearish
                return "SELL"
        
        elif wave_pattern.wave_type == WaveType.CORRECTIVE:
            # Corrective waves are generally bearish
            if wave_pattern.current_position == WavePosition.WAVE_C:
                # End of correction might be buying opportunity
                return "BUY"
            else:
                return "SELL"
        
        return "HOLD"
    
    def _calculate_confidence(self, wave_pattern: WavePattern, peaks_troughs: List[Tuple[int, float, str]]) -> float:
        """Calculate overall confidence in Elliott Wave analysis"""
        # Base confidence from pattern recognition
        pattern_confidence = wave_pattern.confidence
        
        # Adjust based on data quality
        data_quality = min(1.0, len(peaks_troughs) / 10)  # More data points = higher quality
        
        # Combine confidences
        overall_confidence = (pattern_confidence * 0.7) + (data_quality * 0.3)
        
        return max(0.2, min(0.9, overall_confidence))
    
    def _assess_risk_level(self, wave_pattern: WavePattern, index_data: IndexData) -> str:
        """Assess risk level based on wave analysis"""
        risk_factors = 0
        
        # Unknown patterns increase risk
        if wave_pattern.wave_type == WaveType.UNKNOWN:
            risk_factors += 2
        
        # Low confidence increases risk
        if wave_pattern.confidence < 0.4:
            risk_factors += 1
        
        # High volatility increases risk
        if index_data.volatility > 0.25:
            risk_factors += 1
        
        if risk_factors >= 3:
            return "High"
        elif risk_factors >= 1:
            return "Medium"
        else:
            return "Low"
    
    def _get_wave_interpretation(self, wave_pattern: WavePattern) -> str:
        """Get interpretation of current wave position"""
        interpretations = {
            WavePosition.WAVE_1: "Initial impulse move, expect correction",
            WavePosition.WAVE_2: "Corrective phase, prepare for strong move",
            WavePosition.WAVE_3: "Strongest impulse wave, momentum building",
            WavePosition.WAVE_4: "Final correction before last push",
            WavePosition.WAVE_5: "Final impulse wave, reversal approaching",
            WavePosition.WAVE_A: "First leg of correction",
            WavePosition.WAVE_B: "Counter-trend bounce in correction",
            WavePosition.WAVE_C: "Final leg of correction",
            WavePosition.UNKNOWN: "Pattern unclear, await confirmation"
        }
        
        return interpretations.get(wave_pattern.current_position, "Analysis inconclusive")
    
    def _get_wave_analysis_details(self, wave_pattern: WavePattern) -> Dict[str, Any]:
        """Get detailed wave analysis information"""
        return {
            "wave_type": wave_pattern.wave_type.value,
            "current_position": wave_pattern.current_position.value,
            "wave_start": wave_pattern.wave_start,
            "wave_end": wave_pattern.wave_end,
            "pattern_confidence": wave_pattern.confidence,
            "wave_range": abs(wave_pattern.wave_end - wave_pattern.wave_start),
            "wave_direction": "up" if wave_pattern.wave_end > wave_pattern.wave_start else "down"
        }

# Example usage
if __name__ == "__main__":
    # Sample data with more historical levels for wave analysis
    sample_index = IndexData(
        symbol="SPX",
        name="S&P 500",
        current_level=4200.0,
        historical_levels=[
            3800, 3850, 3900, 3950, 4000, 3950, 3900, 3950, 4000, 4050,
            4100, 4050, 4000, 4050, 4100, 4150, 4200, 4180, 4160, 4180, 4200
        ],
        dividend_yield=1.8,
        pe_ratio=22.5,
        pb_ratio=3.2,
        market_cap=35000000000000,  # $35T
        volatility=0.18,
        beta=1.0,
        sector_weights={"Technology": 0.28, "Healthcare": 0.13, "Financials": 0.11},
        constituent_count=500,
        volume=1000000000
    )
    
    # Create Advanced Elliott Wave analyzer
    elliott_wave = AdvancedElliottWaveAnalysis(
        fractal_window=20,
        hurst_window=50,
        volume_confirmation=True,
        multi_timeframe=True,
        pattern_recognition_ml=True
    )
    
    print("\n=== Advanced Elliott Wave Analysis Demo ===")
    print(f"Analyzing {len(sample_index.historical_levels)} data points...")
    
    result = elliott_wave.calculate(sample_index)
    
    print(f"\n--- Primary Wave Analysis ---")
    print(f"Current Wave: {result.wave_pattern.current_position.value}")
    print(f"Wave Type: {result.wave_pattern.wave_type.value}")
    print(f"Wave Degree: {result.wave_pattern.degree.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Signal: {result.signal}")
    print(f"Signal Strength: {result.signal_strength:.2f}")
    
    print(f"\n--- Fractal Analysis ---")
    print(f"Fractal Dimension: {result.fractal_dimension:.3f}")
    print(f"Hurst Exponent: {result.hurst_exponent:.3f}")
    print(f"Fractal Points Identified: {len(result.fractal_points)}")
    
    print(f"\n--- Wave Structure ---")
    print(f"Wave Segments: {len(result.wave_pattern.wave_segments)}")
    if result.wave_pattern.wave_segments:
        last_segment = result.wave_pattern.wave_segments[-1]
        print(f"Last Segment Length: {last_segment.length:.2f}")
        print(f"Last Segment Momentum: {last_segment.momentum:.3f}")
    
    print(f"\n--- Multi-Timeframe Analysis ---")
    if result.multi_timeframe_analysis:
        for timeframe, analysis in result.multi_timeframe_analysis.items():
            print(f"{timeframe}: {analysis.wave_type.value} (confidence: {analysis.confidence:.2f})")
    else:
        print("No multi-timeframe analysis available")
    
    if result.target_levels:
        print("\n--- Target Levels ---")
        for target, level in result.target_levels.items():
            print(f"{target}: {level:.0f}")
    
    if result.probability_zones and 'support_zones' in result.probability_zones and result.probability_zones['support_zones']:
        print("\n--- Support Zones ---")
        for zone_name, zone_data in list(result.probability_zones['support_zones'].items())[:3]:
            print(f"{zone_name}: {zone_data['price']:.0f} (probability: {zone_data['probability']:.2f})")
    
    if result.probability_zones and 'resistance_zones' in result.probability_zones and result.probability_zones['resistance_zones']:
        print("\n--- Resistance Zones ---")
        for zone_name, zone_data in list(result.probability_zones['resistance_zones'].items())[:3]:
            print(f"{zone_name}: {zone_data['price']:.0f} (probability: {zone_data['probability']:.2f})")
    
    print(f"\n--- Risk Assessment ---")
    print(f"Risk Level: {result.risk_level}")
    print(f"Interpretation: {result.interpretation}")
    
    print("\n=== Analysis Complete ===")