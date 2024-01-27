import { useState , useRef} from 'react'
import 'bootstrap/dist/css/bootstrap.css'
import FileInput from  './components/FileInput'
import TextInputWithLable from  './components/TextInputWithLable'
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
    <div>
      <form class="row g-3">

          <div className="mb-3 rawC">
            <label htmlFor="exampleFormControlTextarea1" className="form-label">Input Question to be answered</label>
            <textarea className="form-control" id="exampleFormControlTextarea1" rows="7" w-50 ref={ref}/>
            {isShow && <TextInputWithLable value= {answer}/>}
          </div>
          <button type="button" className="btn btn-primary mb-4" onClick={fetchResponse}>Get Answer</button> 

        </form>
    </div>
  )
}

export default App
