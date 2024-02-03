import React, { useState , useRef} from 'react'
import 'bootstrap/dist/css/bootstrap.css'
import FileInput from  './components/FileInput'
import TextInputWithLable from  './components/TextInputWithLable'
import Dropdown from './components/Dropdown'
import NavBarComp from './components/Navbar'
import SpinnerWithText from './components/SpinnerWithText'
// import './App.css'
import api from './api/api'
function App() {
  const [answer,setAnswer] = useState([]);
  const [isShow,setShow] = useState(false);

  const fetchResponse = async () =>{
    console.log(ref.current.value);
    const question = ref.current.value;
    setShow(true)
    const response = await api.get('/getanswer?question='+question);
    console.log(response.data);
    setAnswer(response.data)
    setShow(false)
  }
  const ref = useRef(null);

  return (
    <React.Fragment>
      
      <NavBarComp />
      <main className='container'>
        <form className="row g-3" >
          
            <div>
              <label htmlFor="inputLable" className="form-label">Input Ad description to be generated</label>
              <textarea className="form-control" id="inputTextarea" rows="7" ref={ref}/>
            </div>

            {isShow && <SpinnerWithText />}

            <button type="button" className="btn btn-primary mb-4" onClick={fetchResponse}>Get Ad</button> 

            <div>
              <TextInputWithLable value= {answer}/>
            </div>

        </form>
      </main>

    </React.Fragment>
  )
}

export default App
