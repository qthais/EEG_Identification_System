interface ProfileCardProps {
    name: string;
    registeredAt: string;
}

export const ProfileCard: React.FC<ProfileCardProps> = ({ name, registeredAt }) => {
    return (
        <div className="bg-gray-50 rounded-lg shadow p-8 w-80 text-center">
            <div className="w-40 h-40 rounded-full bg-gray-300 mx-auto mb-6"></div>
            <h3 className="font-medium text-lg">{name}</h3>
            <p className="text-gray-500 text-sm mt-2">
                Data registered at: {registeredAt}
            </p>
        </div>
    );
}
