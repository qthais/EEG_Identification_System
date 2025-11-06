import { User } from "@/types/User";
import NavBar from "../components/NavBar";
import { ProfileCard } from "../components/ProfileCard";
import { authOptions } from "../api/auth/[...nextauth]/route";
import { getServerSession } from "next-auth";
import { redirect } from "next/navigation";
import ClientOnly from "../components/ClientOnly";
import axios from "axios";
import EEGHistoryViewer, { Prediction } from "../components/EEGHistoryView";

const apiUrl = process.env.NEXT_PUBLIC_API_URL;
export default async function DashboardPage() {
  const session = await getServerSession(authOptions);
  if (!session) {
    redirect("/signin");
  }
  const user: User = {
    name: session.user?.name || "Unknown",
    email: session.user?.email || "unknown@email.com",
  };
  const token = session.accessToken;

  // let rawData: number[][] = [];
  // let confidence: number | null = null;
  // let predictedClass: number | null = null;
  let predictions:Prediction[]=[];
  try {
    const res = await axios.get(`${apiUrl}/api/predictions`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    user.name = res.data.user_name;

    if (res.status === 200 && res.data?.predictions?.length > 0) {
      // Pick the latest prediction
      predictions = res.data.predictions;
      // rawData = latest.raw_data || [];
      // confidence = latest.confidence ?? null;
      // predictedClass = latest.predicted_class ?? null;
    } else {
      console.warn("No predictions found or API returned non-200");
    }
  } catch (err: any) {
    console.error("Error fetching predictions:", err.response?.data || err.message);
  }

  return (
    <>
      <NavBar user={user} />
      <main className="min-h-screen flex flex-col bg-white">
        <div className="flex flex-1 p-10 gap-10">
          <ProfileCard
            name={user.name}
            registeredAt="June 2 2025"
          />

          {/* 👇 Bọc phần client bằng ClientOnly */}
          <ClientOnly>
            <EEGHistoryViewer predictions={predictions || []} />
          </ClientOnly>
        </div>
      </main>
    </>
  );
}
