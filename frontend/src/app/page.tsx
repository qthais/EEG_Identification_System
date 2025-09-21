import { User } from '@/types/User'
import React from 'react'
import NavBar from './components/NavBar'
import Title from './components/HeroSection'
import Hero from './components/HeroSection'

interface LandingPageProps {
  currentUser?: User
}
const LandingPage: React.FC<LandingPageProps> = () => {
  return (
    <div>
      <NavBar />
      <Hero/>
    </div>
  )
}

export default LandingPage