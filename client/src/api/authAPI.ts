import apiClient from "./axios";
import type {
  LoginPayload,
  RegisterPayload,
  RegisteredUser,
  TokenResponse,
  AuthUser,
} from "../types/auth";

export const authApi = {
  /**
   * POST /auth/register
   * Backend expects JSON: { username, email, password }
   * Returns a bare UserResponse (not wrapped in APIResponse).
   */
  register: (payload: RegisterPayload) => {
    return apiClient.post<RegisteredUser>("/auth/register", payload);
  },

  /**
   * POST /auth/login
   * Backend uses OAuth2PasswordRequestForm, so it needs
   * application/x-www-form-urlencoded, NOT JSON — and the form field
   * is called "username" even though we send the user's email in it.
   * Returns a bare TokenResponse (not wrapped in APIResponse).
   */
  login: (payload: LoginPayload) => {
    const form = new URLSearchParams();
    form.append("username", payload.email);
    form.append("password", payload.password);

    return apiClient.post<TokenResponse>("/auth/login", form, {
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
    });
  },

  /**
   * GET /auth/me
   * Requires Authorization: Bearer <token> (attached automatically by the
   * request interceptor in axios.ts). Returns the full profile, including
   * run counts.
   */
  getCurrentUser: () => {
    return apiClient.get<AuthUser>("/auth/me");
  },
};
