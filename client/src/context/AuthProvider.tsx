// src/context/AuthProvider.tsx
import { useState, useEffect, useCallback, type ReactNode } from "react";

import { AuthContext } from "./AuthContext";
import { authApi } from "../api/authAPI";
import { tokenStorage, registerUnauthorizedHandler } from "../api/axios";

import type {
  AuthUser,
  AuthContextValue,
  LoginPayload,
  RegisterPayload,
} from "../types/auth";

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Client-side only — backend has no /auth/logout endpoint.
  const logout = useCallback(() => {
    tokenStorage.clear();
    setUser(null);
  }, []);

  useEffect(() => {
    const bootstrap = async () => {
      const token = tokenStorage.get();

      if (!token) {
        setIsLoading(false);
        return;
      }

      try {
        const res = await authApi.getCurrentUser();
        setUser(res.data); // bare UserProfileResponse, no unwrapping needed
      } catch {
        tokenStorage.clear();
        setUser(null);
      } finally {
        setIsLoading(false);
      }
    };

    bootstrap();
  }, []);

  useEffect(() => {
    registerUnauthorizedHandler(() => {
      tokenStorage.clear();
      setUser(null);
    });
  }, []);

  const login = useCallback(async (payload: LoginPayload) => {
    const tokenRes = await authApi.login(payload);
    tokenStorage.set(tokenRes.data.access_token);

    try {
      const userRes = await authApi.getCurrentUser();
      setUser(userRes.data);
    } catch (err) {
      tokenStorage.clear();
      throw err;
    }
  }, []);

  const register = useCallback(
    async (payload: RegisterPayload) => {
      await authApi.register(payload);
      await login({ email: payload.email, password: payload.password });
    },
    [login],
  );

  const value: AuthContextValue = {
    user,
    isAuthenticated: user !== null,
    isLoading,
    login,
    register,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
