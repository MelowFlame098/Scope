import React from 'react';
import { cn } from '../../utils';

interface SelectProps {
  value?: string;
  onValueChange?: (value: string) => void;
  defaultValue?: string;
  children: React.ReactNode;
}

interface SelectTriggerProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
}

interface SelectContentProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

interface SelectItemProps extends React.HTMLAttributes<HTMLDivElement> {
  value: string;
  children: React.ReactNode;
}

interface SelectValueProps extends React.HTMLAttributes<HTMLSpanElement> {
  placeholder?: string;
}

const SelectContext = React.createContext<{
  value?: string;
  onValueChange?: (value: string) => void;
  open: boolean;
  setOpen: (open: boolean) => void;
}>({ open: false, setOpen: () => {} });

const Select = React.forwardRef<HTMLDivElement, SelectProps>(
  ({ value, onValueChange, defaultValue, children }, ref) => {
    const [internalValue, setInternalValue] = React.useState(defaultValue || '');
    const [open, setOpen] = React.useState(false);
    const currentValue = value !== undefined ? value : internalValue;
    
    const handleValueChange = React.useCallback((newValue: string) => {
      if (value === undefined) {
        setInternalValue(newValue);
      }
      onValueChange?.(newValue);
      setOpen(false);
    }, [value, onValueChange]);

    return (
      <SelectContext.Provider value={{ value: currentValue, onValueChange: handleValueChange, open, setOpen }}>
        <div ref={ref} className="relative">
          {children}
        </div>
      </SelectContext.Provider>
    );
  }
);
Select.displayName = "Select";

const SelectTrigger = React.forwardRef<HTMLButtonElement, SelectTriggerProps>(
  ({ className, children, ...props }, ref) => {
    const context = React.useContext(SelectContext);
    
    return (
      <button
        ref={ref}
        type="button"
        className={cn(
          "flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        onClick={() => context.setOpen(!context.open)}
        {...props}
      >
        {children}
        <svg
          className="h-4 w-4 opacity-50"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>
    );
  }
);
SelectTrigger.displayName = "SelectTrigger";

const SelectValue = React.forwardRef<HTMLSpanElement, SelectValueProps>(
  ({ className, placeholder, ...props }, ref) => {
    const context = React.useContext(SelectContext);
    
    return (
      <span
        ref={ref}
        className={cn("block truncate", className)}
        {...props}
      >
        {context.value || placeholder}
      </span>
    );
  }
);
SelectValue.displayName = "SelectValue";

const SelectContent = React.forwardRef<HTMLDivElement, SelectContentProps>(
  ({ className, children, ...props }, ref) => {
    const context = React.useContext(SelectContext);
    
    if (!context.open) return null;
    
    return (
      <div
        ref={ref}
        className={cn(
          "absolute top-full z-50 mt-1 max-h-60 w-full overflow-auto rounded-md border bg-popover text-popover-foreground shadow-md",
          className
        )}
        {...props}
      >
        {children}
      </div>
    );
  }
);
SelectContent.displayName = "SelectContent";

const SelectItem = React.forwardRef<HTMLDivElement, SelectItemProps>(
  ({ className, children, value, ...props }, ref) => {
    const context = React.useContext(SelectContext);
    const isSelected = context.value === value;
    
    return (
      <div
        ref={ref}
        className={cn(
          "relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none hover:bg-accent hover:text-accent-foreground",
          isSelected && "bg-accent text-accent-foreground",
          className
        )}
        onClick={() => context.onValueChange?.(value)}
        {...props}
      >
        {isSelected && (
          <span className="absolute left-2 flex h-3.5 w-3.5 items-center justify-center">
            <svg
              className="h-4 w-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
          </span>
        )}
        {children}
      </div>
    );
  }
);
SelectItem.displayName = "SelectItem";

export { Select, SelectTrigger, SelectValue, SelectContent, SelectItem };