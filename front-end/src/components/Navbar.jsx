import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import NavDropdown from 'react-bootstrap/NavDropdown';

function NavBarComp() {
  return (
    <Navbar expand="lg" className="bg-body-tertiary container-fluid">
      <Container fluid>
        <Navbar.Brand href="#home">Text Generation for Amharic Language</Navbar.Brand>
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="me-auto">
            <NavDropdown title="Select Model" id="basic-nav-dropdown">
                <NavDropdown.Item href="#/action-1">meta-llama/Llama-2-7b</NavDropdown.Item>
                <NavDropdown.Item href="#/action-2">mistralai/Mixtral-8x7B-v0.1</NavDropdown.Item>
                <NavDropdown.Item href="#/action-3">iocuydi/llama-2-amharic-3784m</NavDropdown.Item>
            </NavDropdown>
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
}

export default NavBarComp;