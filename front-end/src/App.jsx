import { useState , useRef} from 'react'
import 'bootstrap/dist/css/bootstrap.css'
import FileInput from  './components/FileInput'
import TextInputWithLable from  './components/TextInputWithLable'
import Dropdown from './components/Dropdown'
import NavBarComp from './components/Navbar'
import api from './api/api'
function App() {
  const [answer,setAnswer] = useState([]);
  const [isShow,setShow] = useState(false);

  const fetchResponse = async () =>{
    console.log(ref.current.value);
    const question = ref.current.value;

    const response = await api.get('/getanswer?question='+question);
    console.log(response.data);
    setAnswer(response.data)
    setShow(true)
  }
  const ref = useRef(null);

  return (
    <div className='container-fluid'>
      
      <NavBarComp />

      <form class="row g-3" >
         
          <div>
            <label htmlFor="inputLable" className="form-label">Input Question to be answered</label>
            <textarea className="form-control" id="inputTextarea" rows="7" w-50 ref={ref}/>
          </div>
          <div>
            <TextInputWithLable value= {answer}/>
          </div>

         
          <button type="button" className="btn btn-primary mb-4" onClick={fetchResponse}>Get Answer</button> 

      </form>
    </div>
  )
}

export default App
