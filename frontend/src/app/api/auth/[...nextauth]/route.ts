import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import axios from "axios";

const apiUrl = process.env.NEXT_PUBLIC_API_URL!;

const handler = NextAuth({
  providers: [
    CredentialsProvider({
      name: "EEG Login",
      credentials: {
        accessToken: { label: "EEG Token", type: "text" },
      },
      async authorize(credentials) {
        if (!credentials?.accessToken) {
          console.log("❌ Missing token");
          return null;
        }

        // ✅ giả lập user từ token
        return {
          id: "EEG_USER",
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
});

export { handler as GET, handler as POST };
