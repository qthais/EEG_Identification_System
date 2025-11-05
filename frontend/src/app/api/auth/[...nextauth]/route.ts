import NextAuth, { NextAuthOptions } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";

export const authOptions:NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: "EEG Login",
      credentials: {
        accessToken: { label: "EEG Token", type: "text" },
        userId: { label: "User ID", type: "text" },
      },
      async authorize(credentials) {
        if (!credentials?.accessToken) {
          console.log("❌ Missing token");
          return null;
        }

        // ✅ giả lập user từ token
        return {
          id: credentials.userId || "EEG_USER",
          accessToken: credentials.accessToken,
        };
      },
    }),
  ],
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id;
        token.accessToken = user.accessToken;
      }
      return token;
    },
    async session({ session, token }) {
      const { id, accessToken } = token;
      session.user.id = id;
      session.accessToken = accessToken;
      return session;
    },
  },
  secret: process.env.NEXTAUTH_SECRET,
};
const handler = NextAuth(authOptions);

export { handler as GET, handler as POST };
