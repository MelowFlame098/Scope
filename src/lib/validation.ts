// Comprehensive validation utilities for forms and user input

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings?: string[];
}

export interface PasswordStrength {
  score: number; // 0-4 (0: very weak, 4: very strong)
  feedback: string[];
  isValid: boolean;
}

// Email validation
export const validateEmail = (email: string): ValidationResult => {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!email) {
    errors.push('Email is required');
    return { isValid: false, errors, warnings };
  }

  if (email.length > 254) {
    errors.push('Email is too long (maximum 254 characters)');
  }

  // RFC 5322 compliant email regex (simplified)
  const emailRegex = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/;
  
  if (!emailRegex.test(email)) {
    errors.push('Please enter a valid email address');
  }

  // Check for common typos
  const commonDomains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'icloud.com'];
  const domain = email.split('@')[1];
  if (domain) {
    const similarDomain = commonDomains.find(d => 
      levenshteinDistance(domain.toLowerCase(), d) === 1
    );
    if (similarDomain) {
      warnings.push(`Did you mean ${email.split('@')[0]}@${similarDomain}?`);
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
};

// Password validation with strength assessment
export const validatePassword = (password: string): ValidationResult & { strength: PasswordStrength } => {
  const errors: string[] = [];
  const feedback: string[] = [];
  let score = 0;

  if (!password) {
    errors.push('Password is required');
    return {
      isValid: false,
      errors,
      strength: { score: 0, feedback: ['Password is required'], isValid: false }
    };
  }

  // Length check
  if (password.length < 8) {
    errors.push('Password must be at least 8 characters long');
    feedback.push('Use at least 8 characters');
  } else if (password.length >= 8) {
    score += 1;
  }

  if (password.length >= 12) {
    score += 1;
    feedback.push('Good length');
  }

  // Character variety checks
  const hasLowercase = /[a-z]/.test(password);
  const hasUppercase = /[A-Z]/.test(password);
  const hasNumbers = /\d/.test(password);
  const hasSpecialChars = /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password);

  if (!hasLowercase) {
    feedback.push('Add lowercase letters');
  }
  if (!hasUppercase) {
    feedback.push('Add uppercase letters');
  }
  if (!hasNumbers) {
    feedback.push('Add numbers');
  }
  if (!hasSpecialChars) {
    feedback.push('Add special characters (!@#$%^&*)');
  }

  // Score based on character variety
  const varietyScore = [hasLowercase, hasUppercase, hasNumbers, hasSpecialChars]
    .filter(Boolean).length;
  
  if (varietyScore >= 3) {
    score += 1;
  }
  if (varietyScore === 4) {
    score += 1;
  }

  // Check for common patterns
  const commonPatterns = [
    /123456/,
    /password/i,
    /qwerty/i,
    /abc123/i,
    /(.)\1{2,}/, // repeated characters
  ];

  const hasCommonPattern = commonPatterns.some(pattern => pattern.test(password));
  if (hasCommonPattern) {
    feedback.push('Avoid common patterns and repeated characters');
    score = Math.max(0, score - 1);
  }

  // Sequential characters check
  const hasSequential = hasSequentialChars(password);
  if (hasSequential) {
    feedback.push('Avoid sequential characters (abc, 123)');
    score = Math.max(0, score - 1);
  }

  // Final score adjustment
  score = Math.min(4, Math.max(0, score));

  const strengthLabels = ['Very Weak', 'Weak', 'Fair', 'Good', 'Strong'];
  const strengthLabel = strengthLabels[score];

  if (score >= 3) {
    feedback.push(`${strengthLabel} password!`);
  }

  const isPasswordValid = password.length >= 8 && hasLowercase && hasUppercase && 
                         (hasNumbers || hasSpecialChars);

  if (!isPasswordValid && password.length >= 8) {
    errors.push('Password must contain uppercase, lowercase, and either numbers or special characters');
  }

  return {
    isValid: errors.length === 0,
    errors,
    strength: {
      score,
      feedback,
      isValid: isPasswordValid
    }
  };
};

// Name validation
export const validateName = (name: string, fieldName: string = 'Name'): ValidationResult => {
  const errors: string[] = [];

  if (!name || !name.trim()) {
    errors.push(`${fieldName} is required`);
    return { isValid: false, errors };
  }

  const trimmedName = name.trim();

  if (trimmedName.length < 2) {
    errors.push(`${fieldName} must be at least 2 characters long`);
  }

  if (trimmedName.length > 50) {
    errors.push(`${fieldName} must be less than 50 characters`);
  }

  // Allow letters, spaces, hyphens, and apostrophes
  const nameRegex = /^[a-zA-Z\s\-']+$/;
  if (!nameRegex.test(trimmedName)) {
    errors.push(`${fieldName} can only contain letters, spaces, hyphens, and apostrophes`);
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};

// Phone number validation (international format)
export const validatePhoneNumber = (phone: string): ValidationResult => {
  const errors: string[] = [];

  if (!phone) {
    errors.push('Phone number is required');
    return { isValid: false, errors };
  }

  // Remove all non-digit characters for validation
  const digitsOnly = phone.replace(/\D/g, '');

  if (digitsOnly.length < 10) {
    errors.push('Phone number must be at least 10 digits');
  }

  if (digitsOnly.length > 15) {
    errors.push('Phone number must be less than 15 digits');
  }

  // Basic international phone format check
  const phoneRegex = /^[+]?[1-9]\d{1,14}$/;
  if (!phoneRegex.test(digitsOnly)) {
    errors.push('Please enter a valid phone number');
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};

// Date validation
export const validateDate = (date: string, fieldName: string = 'Date'): ValidationResult => {
  const errors: string[] = [];

  if (!date) {
    errors.push(`${fieldName} is required`);
    return { isValid: false, errors };
  }

  const parsedDate = new Date(date);
  
  if (isNaN(parsedDate.getTime())) {
    errors.push(`Please enter a valid ${fieldName.toLowerCase()}`);
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};

// Age validation (for birth date)
export const validateAge = (birthDate: string, minAge: number = 18): ValidationResult => {
  const errors: string[] = [];

  const dateValidation = validateDate(birthDate, 'Birth date');
  if (!dateValidation.isValid) {
    return dateValidation;
  }

  const birth = new Date(birthDate);
  const today = new Date();
  const age = today.getFullYear() - birth.getFullYear();
  const monthDiff = today.getMonth() - birth.getMonth();
  
  const actualAge = monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate()) 
    ? age - 1 
    : age;

  if (actualAge < minAge) {
    errors.push(`You must be at least ${minAge} years old`);
  }

  if (actualAge > 120) {
    errors.push('Please enter a valid birth date');
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};

// URL validation
export const validateUrl = (url: string, fieldName: string = 'URL'): ValidationResult => {
  const errors: string[] = [];

  if (!url) {
    errors.push(`${fieldName} is required`);
    return { isValid: false, errors };
  }

  try {
    new URL(url);
  } catch {
    errors.push(`Please enter a valid ${fieldName.toLowerCase()}`);
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};

// Generic required field validation
export const validateRequired = (value: any, fieldName: string): ValidationResult => {
  const errors: string[] = [];

  if (value === null || value === undefined || 
      (typeof value === 'string' && !value.trim()) ||
      (Array.isArray(value) && value.length === 0)) {
    errors.push(`${fieldName} is required`);
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};

// Helper functions
function levenshteinDistance(str1: string, str2: string): number {
  const matrix = [];

  for (let i = 0; i <= str2.length; i++) {
    matrix[i] = [i];
  }

  for (let j = 0; j <= str1.length; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= str2.length; i++) {
    for (let j = 1; j <= str1.length; j++) {
      if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }

  return matrix[str2.length][str1.length];
}

function hasSequentialChars(password: string): boolean {
  const sequences = [
    'abcdefghijklmnopqrstuvwxyz',
    '0123456789',
    'qwertyuiopasdfghjklzxcvbnm'
  ];

  const lowerPassword = password.toLowerCase();
  
  for (const sequence of sequences) {
    for (let i = 0; i <= sequence.length - 3; i++) {
      const subseq = sequence.substring(i, i + 3);
      if (lowerPassword.includes(subseq) || lowerPassword.includes(subseq.split('').reverse().join(''))) {
        return true;
      }
    }
  }
  
  return false;
}

// Form validation helper
export const validateForm = (data: Record<string, any>, rules: Record<string, (value: any) => ValidationResult>): {
  isValid: boolean;
  errors: Record<string, string[]>;
  warnings?: Record<string, string[]>;
} => {
  const errors: Record<string, string[]> = {};
  const warnings: Record<string, string[]> = {};
  let isValid = true;

  for (const [field, validator] of Object.entries(rules)) {
    const result = validator(data[field]);
    
    if (!result.isValid) {
      errors[field] = result.errors;
      isValid = false;
    }
    
    if (result.warnings && result.warnings.length > 0) {
      warnings[field] = result.warnings;
    }
  }

  return {
    isValid,
    errors,
    warnings: Object.keys(warnings).length > 0 ? warnings : undefined
  };
};

// Export commonly used validation rules
export const validationRules = {
  email: validateEmail,
  password: validatePassword,
  firstName: (value: string) => validateName(value, 'First name'),
  lastName: (value: string) => validateName(value, 'Last name'),
  phoneNumber: validatePhoneNumber,
  birthDate: validateDate,
  age: validateAge,
  url: validateUrl,
  required: (fieldName: string) => (value: any) => validateRequired(value, fieldName),
};