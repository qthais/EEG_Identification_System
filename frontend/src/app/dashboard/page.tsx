import { User } from "@/types/User";
import EEGChart from "../components/Chart";
import NavBar from "../components/NavBar";
import { ProfileCard } from "../components/ProfileCard";
import { authOptions } from "../api/auth/[...nextauth]/route";
import { getServerSession } from "next-auth";
import { redirect } from "next/navigation";


export default async function DashboardPage() {
    const session = await getServerSession(authOptions)
    if (!session) {
        redirect("/signin"); // 🔒 redirect if not logged in
    }
    const user:User={
        name: "Thai",
        email:"thai@gmail.com"
    }
    return (
        <>
            <NavBar user={user}  />
            <main className="min-h-screen flex flex-col bg-white">

                <div className="flex flex-1 p-10 gap-10">
                    {/* Left: Profile */}
                    <ProfileCard
                        name="Nguyen Huu Quoc Thai"
                        registeredAt="June 2 2025"
                    />

                    {/* Right: Model & Chart */}
                    <div className="flex-1">
                        <h2 className="text-lg font-bold mb-6">
                            MODEL CONFIDENCE: <span className="text-black">95%</span>
                        </h2>
                        <EEGChart />
                    </div>
                </div>
            </main>
        </>
    );
}
