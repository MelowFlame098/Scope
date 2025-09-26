import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { AuthService } from '../../services/AuthService';
import { BiometricService } from '../../services/BiometricService';
import { StorageService } from '../../services/StorageService';

export interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  avatar?: string;
  role: 'user' | 'premium' | 'institutional';
  preferences: {
    theme: 'light' | 'dark' | 'auto';
    currency: string;
    language: string;
    notifications: {
      push: boolean;
      email: boolean;
      sms: boolean;
      priceAlerts: boolean;
      newsAlerts: boolean;
      tradingAlerts: boolean;
    };
  };
  subscription: {
    plan: 'free' | 'premium' | 'enterprise';
    expiresAt?: string;
    features: string[];
  };
  security: {
    twoFactorEnabled: boolean;
    biometricEnabled: boolean;
    lastLogin: string;
    loginAttempts: number;
  };
}

export interface AuthState {
  user: User | null;
  token: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  biometricAvailable: boolean;
  biometricEnabled: boolean;
  loginAttempts: number;
  isLocked: boolean;
  lockUntil: number | null;
}

const initialState: AuthState = {
  user: null,
  token: null,
  refreshToken: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,
  biometricAvailable: false,
  biometricEnabled: false,
  loginAttempts: 0,
  isLocked: false,
  lockUntil: null,
};

// Async thunks
export const login = createAsyncThunk(
  'auth/login',
  async (credentials: { email: string; password: string }, { rejectWithValue }) => {
    try {
      const response = await AuthService.login(credentials);
      await StorageService.setSecureItem('token', response.token);
      await StorageService.setSecureItem('refreshToken', response.refreshToken);
      return response;
    } catch (error: any) {
      return rejectWithValue(error.message || 'Login failed');
    }
  }
);

export const loginWithBiometric = createAsyncThunk(
  'auth/loginWithBiometric',
  async (_, { rejectWithValue }) => {
    try {
      const isAvailable = await BiometricService.isAvailable();
      if (!isAvailable) {
        throw new Error('Biometric authentication not available');
      }

      const result = await BiometricService.authenticate('Please authenticate to access your account');
      if (!result.success) {
        throw new Error(result.error || 'Biometric authentication failed');
      }

      const storedToken = await StorageService.getSecureItem('token');
      const storedRefreshToken = await StorageService.getSecureItem('refreshToken');
      
      if (!storedToken) {
        throw new Error('No stored credentials found');
      }

      const user = await AuthService.validateToken(storedToken);
      return {
        user,
        token: storedToken,
        refreshToken: storedRefreshToken,
      };
    } catch (error: any) {
      return rejectWithValue(error.message || 'Biometric login failed');
    }
  }
);

export const register = createAsyncThunk(
  'auth/register',
  async (userData: {
    email: string;
    password: string;
    firstName: string;
    lastName: string;
  }, { rejectWithValue }) => {
    try {
      const response = await AuthService.register(userData);
      await StorageService.setSecureItem('token', response.token);
      await StorageService.setSecureItem('refreshToken', response.refreshToken);
      return response;
    } catch (error: any) {
      return rejectWithValue(error.message || 'Registration failed');
    }
  }
);

export const refreshToken = createAsyncThunk(
  'auth/refreshToken',
  async (_, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: AuthState };
      const refreshToken = state.auth.refreshToken;
      
      if (!refreshToken) {
        throw new Error('No refresh token available');
      }

      const response = await AuthService.refreshToken(refreshToken);
      await StorageService.setSecureItem('token', response.token);
      await StorageService.setSecureItem('refreshToken', response.refreshToken);
      return response;
    } catch (error: any) {
      return rejectWithValue(error.message || 'Token refresh failed');
    }
  }
);

export const logout = createAsyncThunk(
  'auth/logout',
  async (_, { getState }) => {
    try {
      const state = getState() as { auth: AuthState };
      if (state.auth.token) {
        await AuthService.logout(state.auth.token);
      }
    } catch (error) {
      // Continue with logout even if API call fails
      console.warn('Logout API call failed:', error);
    } finally {
      await StorageService.removeSecureItem('token');
      await StorageService.removeSecureItem('refreshToken');
    }
  }
);

export const updateProfile = createAsyncThunk(
  'auth/updateProfile',
  async (updates: Partial<User>, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: AuthState };
      if (!state.auth.token) {
        throw new Error('Not authenticated');
      }

      const updatedUser = await AuthService.updateProfile(updates, state.auth.token);
      return updatedUser;
    } catch (error: any) {
      return rejectWithValue(error.message || 'Profile update failed');
    }
  }
);

export const enableBiometric = createAsyncThunk(
  'auth/enableBiometric',
  async (_, { rejectWithValue }) => {
    try {
      const isAvailable = await BiometricService.isAvailable();
      if (!isAvailable) {
        throw new Error('Biometric authentication not available');
      }

      const result = await BiometricService.authenticate('Enable biometric authentication for secure access');
      if (!result.success) {
        throw new Error(result.error || 'Biometric setup failed');
      }

      await StorageService.setItem('biometricEnabled', 'true');
      return true;
    } catch (error: any) {
      return rejectWithValue(error.message || 'Biometric setup failed');
    }
  }
);

export const disableBiometric = createAsyncThunk(
  'auth/disableBiometric',
  async () => {
    await StorageService.removeItem('biometricEnabled');
    return false;
  }
);

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    incrementLoginAttempts: (state) => {
      state.loginAttempts += 1;
      if (state.loginAttempts >= 5) {
        state.isLocked = true;
        state.lockUntil = Date.now() + 15 * 60 * 1000; // 15 minutes
      }
    },
    resetLoginAttempts: (state) => {
      state.loginAttempts = 0;
      state.isLocked = false;
      state.lockUntil = null;
    },
    checkLockStatus: (state) => {
      if (state.lockUntil && Date.now() > state.lockUntil) {
        state.isLocked = false;
        state.lockUntil = null;
        state.loginAttempts = 0;
      }
    },
    setBiometricAvailable: (state, action: PayloadAction<boolean>) => {
      state.biometricAvailable = action.payload;
    },
    updateUserPreferences: (state, action: PayloadAction<Partial<User['preferences']>>) => {
      if (state.user) {
        state.user.preferences = { ...state.user.preferences, ...action.payload };
      }
    },
  },
  extraReducers: (builder) => {
    // Login
    builder
      .addCase(login.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(login.fulfilled, (state, action) => {
        state.isLoading = false;
        state.isAuthenticated = true;
        state.user = action.payload.user;
        state.token = action.payload.token;
        state.refreshToken = action.payload.refreshToken;
        state.loginAttempts = 0;
        state.isLocked = false;
        state.lockUntil = null;
      })
      .addCase(login.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
        state.loginAttempts += 1;
        if (state.loginAttempts >= 5) {
          state.isLocked = true;
          state.lockUntil = Date.now() + 15 * 60 * 1000;
        }
      })

    // Biometric Login
    builder
      .addCase(loginWithBiometric.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(loginWithBiometric.fulfilled, (state, action) => {
        state.isLoading = false;
        state.isAuthenticated = true;
        state.user = action.payload.user;
        state.token = action.payload.token;
        state.refreshToken = action.payload.refreshToken;
      })
      .addCase(loginWithBiometric.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })

    // Register
    builder
      .addCase(register.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(register.fulfilled, (state, action) => {
        state.isLoading = false;
        state.isAuthenticated = true;
        state.user = action.payload.user;
        state.token = action.payload.token;
        state.refreshToken = action.payload.refreshToken;
      })
      .addCase(register.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })

    // Refresh Token
    builder
      .addCase(refreshToken.fulfilled, (state, action) => {
        state.token = action.payload.token;
        state.refreshToken = action.payload.refreshToken;
      })
      .addCase(refreshToken.rejected, (state) => {
        state.isAuthenticated = false;
        state.user = null;
        state.token = null;
        state.refreshToken = null;
      })

    // Logout
    builder
      .addCase(logout.fulfilled, (state) => {
        state.isAuthenticated = false;
        state.user = null;
        state.token = null;
        state.refreshToken = null;
        state.error = null;
      })

    // Update Profile
    builder
      .addCase(updateProfile.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(updateProfile.fulfilled, (state, action) => {
        state.isLoading = false;
        state.user = action.payload;
      })
      .addCase(updateProfile.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })

    // Biometric
    builder
      .addCase(enableBiometric.fulfilled, (state) => {
        state.biometricEnabled = true;
      })
      .addCase(disableBiometric.fulfilled, (state) => {
        state.biometricEnabled = false;
      });
  },
});

export const {
  clearError,
  incrementLoginAttempts,
  resetLoginAttempts,
  checkLockStatus,
  setBiometricAvailable,
  updateUserPreferences,
} = authSlice.actions;

export default authSlice.reducer;