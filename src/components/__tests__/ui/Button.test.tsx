import { render, screen, fireEvent } from '@testing-library/react'
import { Button } from '../../ui/button'

describe('Button Component', () => {
  it('renders button with text', () => {
    render(<Button>Click me</Button>)
    
    const button = screen.getByRole('button', { name: /click me/i })
    expect(button).toBeInTheDocument()
  })

  it('handles click events', () => {
    const handleClick = jest.fn()
    render(<Button onClick={handleClick}>Click me</Button>)
    
    const button = screen.getByRole('button', { name: /click me/i })
    fireEvent.click(button)
    
    expect(handleClick).toHaveBeenCalledTimes(1)
  })

  it('applies variant classes correctly', () => {
    render(<Button variant="destructive">Delete</Button>)
    
    const button = screen.getByRole('button', { name: /delete/i })
    expect(button).toHaveClass('bg-destructive')
  })

  it('applies size classes correctly', () => {
    render(<Button size="lg">Large Button</Button>)
    
    const button = screen.getByRole('button', { name: /large button/i })
    expect(button).toHaveClass('h-11')
  })

  it('handles disabled state', () => {
    render(<Button disabled>Disabled Button</Button>)
    
    const button = screen.getByRole('button', { name: /disabled button/i })
    expect(button).toBeDisabled()
    expect(button).toHaveClass('disabled:pointer-events-none')
  })

  it('renders as child component when asChild is true', () => {
    render(
      <Button asChild>
        <a href="/test">Link Button</a>
      </Button>
    )
    
    const link = screen.getByRole('link', { name: /link button/i })
    expect(link).toBeInTheDocument()
    expect(link).toHaveAttribute('href', '/test')
  })

  it('applies custom className', () => {
    render(<Button className="custom-class">Custom Button</Button>)
    
    const button = screen.getByRole('button', { name: /custom button/i })
    expect(button).toHaveClass('custom-class')
  })

  it('forwards ref correctly', () => {
    const ref = { current: null }
    render(<Button ref={ref}>Ref Button</Button>)
    
    expect(ref.current).toBeInstanceOf(HTMLButtonElement)
  })

  it('handles loading state with spinner', () => {
    render(<Button disabled>Loading...</Button>)
    
    const button = screen.getByRole('button', { name: /loading/i })
    expect(button).toBeDisabled()
  })

  it('supports different button types', () => {
    render(<Button type="submit">Submit</Button>)
    
    const button = screen.getByRole('button', { name: /submit/i })
    expect(button).toHaveAttribute('type', 'submit')
  })
})