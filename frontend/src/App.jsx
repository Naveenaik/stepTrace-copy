import { BrowserRouter,Routes,Route } from "react-router-dom"
import About from "./pages/About"
import Dashboard from "./pages/Dashboard"
import ScrollToTop from "./components/ScrollToTop"
import Excel from "./components/Excel/Excel"
import Recodered from "./Model2/Recodered"
function App() {

  return (
    <>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<About/>}></Route>
        </Routes>
        <Routes>
          <Route path="/dashboard" element={<Dashboard/>}></Route>
        </Routes>
        <Routes>
          <Route path="/excel" element={<Excel/>}></Route>
        </Routes>
        <Routes>
          <Route path="/model2" element={<Recodered/>}></Route>
        </Routes>
      </BrowserRouter>
      <ScrollToTop />
    </>
  )
}

export default App
