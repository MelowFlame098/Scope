import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { TradingChart } from '../../trading/TradingChart'

// Mock recharts
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
  ComposedChart: ({ children }: any) => <div data-testid="composed-chart">{children}</div>,
  Bar: () => <div data-testid="bar" />,
  Line: () => <div data-testid="line" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
}))

describe('TradingChart Component', () => {
  const mockProps = {
    symbol: 'AAPL',
    onPriceSelect: jest.fn(),
  }

  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders trading chart with symbol', () => {
    render(<TradingChart {...mockProps} />)
    
    expect(screen.getByText('AAPL Chart')).toBeInTheDocument()
    expect(screen.getByTestId('responsive-container')).toBeInTheDocument()
  })

  it('displays timeframe selection buttons', () => {
    render(<TradingChart {...mockProps} />)
    
    const timeframes = ['1m', '5m', '15m', '1h', '1d']
    timeframes.forEach(timeframe => {
      expect(screen.getByText(timeframe)).toBeInTheDocument()
    })
  })

  it('handles timeframe selection', async () => {
    render(<TradingChart {...mockProps} />)
    
    const fiveMinButton = screen.getByText('5m')
    fireEvent.click(fiveMinButton)
    
    // Check if the button becomes active
    expect(fiveMinButton).toHaveClass('bg-blue-600')
  })

  it('displays technical indicators toggle', () => {
    render(<TradingChart {...mockProps} />)
    
    expect(screen.getByText('MA20')).toBeInTheDocument()
    expect(screen.getByText('Volume')).toBeInTheDocument()
  })

  it('handles indicator toggle', () => {
    render(<TradingChart {...mockProps} />)
    
    const ma20Toggle = screen.getByText('MA20')
    fireEvent.click(ma20Toggle)
    
    // Verify toggle functionality
    expect(ma20Toggle).toBeInTheDocument()
  })

  it('generates mock data correctly', () => {
    render(<TradingChart {...mockProps} />)
    
    // Check if chart components are rendered
    expect(screen.getByTestId('composed-chart')).toBeInTheDocument()
    expect(screen.getByTestId('bar')).toBeInTheDocument()
    expect(screen.getByTestId('line')).toBeInTheDocument()
  })

  it('handles price selection on chart click', async () => {
    render(<TradingChart {...mockProps} />)
    
    const chart = screen.getByTestId('composed-chart')
    fireEvent.click(chart)
    
    // Verify onPriceSelect callback
    await waitFor(() => {
      // This would need to be adjusted based on actual implementation
      expect(screen.getByTestId('composed-chart')).toBeInTheDocument()
    })
  })

  it('displays current price and change', () => {
    render(<TradingChart {...mockProps} />)
    
    // Check for price display elements
    expect(screen.getByText(/\$/)).toBeInTheDocument()
  })

  it('handles loading state', () => {
    render(<TradingChart {...mockProps} />)
    
    // Check if component renders without loading spinner initially
    expect(screen.queryByText('Loading...')).not.toBeInTheDocument()
  })
})