// Mirrors server/schemas/auth.py and server/schemas/user_profile.py exactly.

export interface RegisterPayload {
  username: string;
  email: string;
  password: string;
}

// Backend's /auth/login expects x-www-form-urlencoded (OAuth2PasswordRequestForm),
// with `username` actually holding the user's email. Kept separate from the
// wire format so the rest of the app can just think in terms of "email".
export interface LoginPayload {
  email: string;
  password: string;
}

// Bare response from POST /auth/register (server/schemas/auth.py: UserResponse)
export interface RegisteredUser {
  id: string;
  username: string;
  email: string;
}

// Bare response from POST /auth/login (server/schemas/auth.py: TokenResponse)
export interface TokenResponse {
  access_token: string;
  token_type: string;
}

// Response from GET /auth/me (server/schemas/user_profile.py: UserProfileResponse)
export interface AuthUser {
  id: string;
  username: string;
  email: string;
  created_at: string;
  total_runs: number;
  successful_runs: number;
  failed_runs: number;
}

// FastAPI's default HTTPException error shape: { "detail": "..." }
// (auth routes raise plain HTTPException, so this is NOT the APIResponse envelope)
export interface ApiErrorShape {
  detail: string;
}

export interface AuthContextValue {
  user: AuthUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (payload: LoginPayload) => Promise<void>;
  register: (payload: RegisterPayload) => Promise<void>;
  logout: () => void;
}
