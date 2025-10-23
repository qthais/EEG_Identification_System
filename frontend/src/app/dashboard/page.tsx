import { User } from "@/types/User";
import NavBar from "../components/NavBar";
import { ProfileCard } from "../components/ProfileCard";
import { authOptions } from "../api/auth/[...nextauth]/route";
import { getServerSession } from "next-auth";
import { redirect } from "next/navigation";
import ClientOnly from "../components/ClientOnly";
import EEGDashboardClient from "./DashboardClient";

export default async function DashboardPage() {
  const session = await getServerSession(authOptions);
  if (!session) {
    redirect("/signin");
  }

  const user: User = {
    name: "Thai",
    email: "thai@gmail.com",
  };

  return (
    <>
      <NavBar user={user} />
      <main className="min-h-screen flex flex-col bg-white">
        <div className="flex flex-1 p-10 gap-10">
          <ProfileCard
            name="Nguyen Huu Quoc Thai"
            registeredAt="June 2 2025"
          />

          {/* 👇 Bọc phần client bằng ClientOnly */}
          <ClientOnly>
            <EEGDashboardClient
            />
          </ClientOnly>
        </div>
      </main>
    </>
  );
}
